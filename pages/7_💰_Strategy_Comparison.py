import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(
    page_title="Investment Strategy Comparison",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
VND_TO_USD = 26000

# DEBUG MODE - Set to True to show only Debug Table section
DEBUG_MODE = False

# @st.cache_data  # DISABLED FOR DEBUG
def get_available_assets(db_path="db/investment_data.db"):
    """Get available cryptocurrencies and VN funds"""
    conn = sqlite3.connect(db_path)
    
    # Get crypto assets
    crypto_query = """
        SELECT DISTINCT asset_code, asset_type
        FROM assets
        WHERE asset_type = 'crypto'
        ORDER BY asset_code
    """
    crypto_df = pd.read_sql_query(crypto_query, conn)
    
    # Get VN funds
    fund_query = """
        SELECT DISTINCT asset_code, asset_type
        FROM assets
        WHERE asset_type = 'vn_fund'
        ORDER BY asset_code
    """
    fund_df = pd.read_sql_query(fund_query, conn)
    
    conn.close()
    
    return crypto_df['asset_code'].tolist(), fund_df['asset_code'].tolist()

# @st.cache_data  # DISABLED FOR DEBUG
def get_price_data(asset_codes, db_path="db/investment_data.db"):
    """Get price data for specified assets"""
    conn = sqlite3.connect(db_path)
    
    placeholders = ','.join(['?' for _ in asset_codes])
    query = f"""
        SELECT asset_code, date, price
        FROM price_data
        WHERE asset_code IN ({placeholders})
        ORDER BY asset_code, date
    """
    
    df = pd.read_sql_query(query, conn, params=asset_codes)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_strategy_1(crypto_data, fund_data, initial_amount, loan_pct, interest_rate, start_date, end_date=None):
    """
    Calculate Strategy 1: Leveraged Crypto + VN Fund
    
    Returns DataFrame with columns: date, crypto_value, fund_value, debt, total_value
    """
    # Get crypto price at start date
    crypto_start = crypto_data[crypto_data['date'] >= start_date].iloc[0]
    crypto_start_price = crypto_start['price']
    crypto_start_date = crypto_start['date']
    
    # Calculate crypto holdings (number of coins/tokens)
    crypto_holdings = initial_amount / crypto_start_price
    
    # Calculate loan amount
    loan_amount = initial_amount * (loan_pct / 100)
    
    # Get fund price at start date (or closest date after)
    fund_start = fund_data[fund_data['date'] >= crypto_start_date]
    if fund_start.empty:
        return pd.DataFrame()
    
    fund_start = fund_start.iloc[0]
    fund_start_price = fund_start['price']
    fund_start_date = fund_start['date']
    
    # Calculate fund holdings (convert VND price to USD)
    fund_price_usd = fund_start_price / VND_TO_USD
    fund_holdings = loan_amount / fund_price_usd
    
    # Merge crypto and fund data on date
    crypto_subset = crypto_data[crypto_data['date'] >= crypto_start_date].copy()
    fund_subset = fund_data[fund_data['date'] >= fund_start_date].copy()
    
    # Filter by end date if provided
    if end_date:
        crypto_subset = crypto_subset[crypto_subset['date'] <= end_date]
        fund_subset = fund_subset[fund_subset['date'] <= end_date]
    
    # Merge on nearest dates
    merged = pd.merge_asof(
        crypto_subset.sort_values('date'),
        fund_subset.sort_values('date'),
        on='date',
        direction='nearest',
        suffixes=('_crypto', '_fund')
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Calculate portfolio components
    merged['crypto_value'] = crypto_holdings * merged['price_crypto']
    merged['fund_value'] = fund_holdings * (merged['price_fund'] / VND_TO_USD)
    
    # Calculate accumulated debt (compound annually)
    merged['years_elapsed'] = (merged['date'] - crypto_start_date).dt.days / 365.25
    merged['debt'] = loan_amount * ((1 + interest_rate / 100) ** merged['years_elapsed'])
    
    # Total portfolio value
    merged['total_value'] = merged['crypto_value'] + merged['fund_value'] - merged['debt']
    
    # Calculate liquidation price (Health Factor = 1.0)
    # Liquidation Price = Loan / (Crypto Holdings × Liquidation Threshold)
    # With LT = 80% and HF = 1.0
    LIQUIDATION_THRESHOLD = 0.80
    merged['liquidation_price'] = merged['debt'] / (crypto_holdings * LIQUIDATION_THRESHOLD)
    
    # Store initial value for CAGR calculation
    merged['initial_value'] = initial_amount
    merged['start_date'] = crypto_start_date
    
    return merged[['date', 'crypto_value', 'fund_value', 'debt', 'total_value', 'price_crypto', 'liquidation_price', 'initial_value', 'start_date']]

def calculate_strategy_2_dca(crypto_data, fund_data, initial_amount, loan_pct, interest_rate, start_date, end_date=None):
    """
    Calculate Strategy 2: DCA Crypto-Backed VN Fund
    - Split initial investment into 12 monthly BTC purchases
    - After each BTC purchase, borrow loan_pct% of that purchase amount to invest in VN fund
    - Cash (uninvested amount) is tracked and included in total_value
    
    Returns DataFrame with columns: date, crypto_value, fund_value, debt, cash, total_value
    """
    monthly_investment = initial_amount / 12
    
    # Get crypto data from start date
    crypto_subset = crypto_data[crypto_data['date'] >= start_date].copy()
    if crypto_subset.empty:
        return pd.DataFrame()
    
    crypto_start_date = crypto_subset.iloc[0]['date']
    
    # Track each monthly purchase with cumulative values
    purchase_records = []
    cumulative_btc = 0
    cumulative_fund_units = 0
    
    for month_idx in range(12):
        # Calculate purchase date (approximately 30 days apart)
        purchase_date = crypto_start_date + timedelta(days=month_idx * 30)
        
        # Find crypto price at purchase date
        crypto_at_purchase = crypto_data[crypto_data['date'] >= purchase_date]
        if crypto_at_purchase.empty:
            break
        
        crypto_price = crypto_at_purchase.iloc[0]['price']
        actual_purchase_date = crypto_at_purchase.iloc[0]['date']
        
        # Buy BTC with monthly investment
        btc_purchased = monthly_investment / crypto_price
        cumulative_btc += btc_purchased
        
        # Calculate loan amount
        loan_amount = monthly_investment * (loan_pct / 100)
        
        # Find fund price at purchase date
        fund_at_purchase = fund_data[fund_data['date'] >= actual_purchase_date]
        if fund_at_purchase.empty:
            continue
        
        fund_price = fund_at_purchase.iloc[0]['price']
        fund_price_usd = fund_price / VND_TO_USD
        
        # Buy VN fund with loan amount
        fund_purchased = loan_amount / fund_price_usd
        cumulative_fund_units += fund_purchased
        
        # Record this purchase with cumulative values
        purchase_records.append({
            'date': actual_purchase_date,
            'month_idx': month_idx + 1,
            'loan_amount': loan_amount,
            'btc_purchased': btc_purchased,
            'btc_price': crypto_price,
            'fund_purchased': fund_purchased,
            'cumulative_btc': cumulative_btc,
            'cumulative_fund_units': cumulative_fund_units,
            'cumulative_invested': monthly_investment * (month_idx + 1)
        })
    
    if not purchase_records:
        return pd.DataFrame()
    
    # Get all dates from first purchase to end
    first_purchase_date = purchase_records[0]['date']
    all_dates = crypto_data[crypto_data['date'] >= first_purchase_date].copy()
    
    if end_date:
        all_dates = all_dates[all_dates['date'] <= end_date]
    
    # Merge with fund data
    fund_subset = fund_data[fund_data['date'] >= first_purchase_date].copy()
    if end_date:
        fund_subset = fund_subset[fund_subset['date'] <= end_date]
    
    merged = pd.merge_asof(
        all_dates.sort_values('date'),
        fund_subset.sort_values('date'),
        on='date',
        direction='nearest',
        suffixes=('_crypto', '_fund')
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Calculate portfolio components for each date with PROGRESSIVE holdings
    results = []
    for _, row in merged.iterrows():
        current_date = row['date']
        current_btc_price = row['price_crypto']
        current_fund_price = row['price_fund']
        
        # Find how many DCA purchases have been completed by this date
        completed_purchases = [p for p in purchase_records if p['date'] <= current_date]
        
        if completed_purchases:
            latest_purchase = completed_purchases[-1]
            current_btc_holdings = latest_purchase['cumulative_btc']
            current_fund_units = latest_purchase['cumulative_fund_units']
            invested_amount = latest_purchase['cumulative_invested']
            dca_months_completed = latest_purchase['month_idx']
        else:
            current_btc_holdings = 0
            current_fund_units = 0
            invested_amount = 0
            dca_months_completed = 0
        
        # Cash = uninvested amount
        cash = initial_amount - invested_amount
        
        # Crypto value at current price
        crypto_value = current_btc_holdings * current_btc_price
        
        # Fund value at current price (convert VND to USD)
        fund_value = current_fund_units * (current_fund_price / VND_TO_USD)
        
        # Debt with interest (each loan compounds from its own date)
        debt = 0
        for p in completed_purchases:
            years_elapsed = (current_date - p['date']).days / 365.25
            if years_elapsed >= 0:
                debt += p['loan_amount'] * ((1 + interest_rate / 100) ** years_elapsed)
        
        # Total value includes cash
        total_value = crypto_value + fund_value - debt + cash
        
        results.append({
            'date': current_date,
            'crypto_value': crypto_value,
            'fund_value': fund_value,
            'debt': debt,
            'cash': cash,
            'total_value': total_value,
            'price_crypto': current_btc_price,
            'btc_holdings': current_btc_holdings,
            'dca_months_completed': dca_months_completed
        })
    
    result_df = pd.DataFrame(results)
    
    # Calculate average BTC buy price (weighted average) after all purchases
    if purchase_records:
        total_btc_cost = sum([p['btc_purchased'] * p['btc_price'] for p in purchase_records])
        final_btc_holdings = purchase_records[-1]['cumulative_btc']
        avg_btc_price = total_btc_cost / final_btc_holdings if final_btc_holdings > 0 else 0
        result_df['avg_btc_buy_price'] = avg_btc_price
    else:
        result_df['avg_btc_buy_price'] = 0
    
    # Store initial value for CAGR calculation
    result_df['initial_value'] = initial_amount
    result_df['start_date'] = first_purchase_date
    
    return result_df[['date', 'crypto_value', 'fund_value', 'debt', 'cash', 'total_value', 'price_crypto', 'avg_btc_buy_price', 'initial_value', 'start_date']]

def calculate_strategy_3(fund_data, initial_amount, start_date, end_date=None):
    """
    Calculate Strategy 2: Simple VN Fund Investment
    
    Returns DataFrame with columns: date, total_value
    """
    # Get fund price at start date
    fund_start = fund_data[fund_data['date'] >= start_date]
    if fund_start.empty:
        return pd.DataFrame()
    
    fund_start = fund_start.iloc[0]
    fund_start_price = fund_start['price']
    fund_start_date = fund_start['date']
    
    # Calculate fund holdings (convert VND price to USD)
    fund_price_usd = fund_start_price / VND_TO_USD
    fund_holdings = initial_amount / fund_price_usd
    
    # Get fund data from start date onwards
    fund_subset = fund_data[fund_data['date'] >= fund_start_date].copy()
    
    # Filter by end date if provided
    if end_date:
        fund_subset = fund_subset[fund_subset['date'] <= end_date]
    
    # Calculate portfolio value
    fund_subset['total_value'] = fund_holdings * (fund_subset['price'] / VND_TO_USD)
    fund_subset['initial_value'] = initial_amount
    fund_subset['start_date'] = fund_start_date
    
    return fund_subset[['date', 'total_value', 'initial_value', 'start_date']]

def calculate_yearly_drawdown(data):
    """
    Calculate maximum drawdown for each year
    
    Args:
        data: DataFrame with 'date' and 'total_value' columns
    
    Returns:
        DataFrame with 'year' and 'max_drawdown' columns
    """
    if data.empty:
        return pd.DataFrame()
    
    data = data.copy()
    data['year'] = pd.to_datetime(data['date']).dt.year
    
    yearly_drawdowns = []
    
    for year in data['year'].unique():
        year_data = data[data['year'] == year].copy()
        
        if len(year_data) < 2:
            continue
        
        # Calculate running maximum (peak)
        year_data['running_max'] = year_data['total_value'].cummax()
        
        # Calculate drawdown from peak
        year_data['drawdown'] = (year_data['total_value'] - year_data['running_max']) / year_data['running_max'] * 100
        
        # Get maximum drawdown (most negative value)
        max_drawdown = year_data['drawdown'].min()
        
        yearly_drawdowns.append({
            'year': int(year),
            'max_drawdown': max_drawdown
        })
    
    return pd.DataFrame(yearly_drawdowns)

def calculate_monthly_returns(data):
    """
    Calculate monthly returns from strategy data
    
    Args:
        data: DataFrame with 'date' and 'total_value' columns
    
    Returns:
        DataFrame with 'year', 'month', 'month_return' columns
    """
    if data.empty:
        return pd.DataFrame()
    
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    
    # Get month-end values
    data['year_month'] = data['date'].dt.to_period('M')
    monthly_data = data.groupby('year_month').agg({
        'total_value': 'last',
        'date': 'last'
    }).reset_index()
    
    # Calculate monthly returns
    monthly_data['month_return'] = monthly_data['total_value'].pct_change() * 100
    monthly_data['year'] = monthly_data['year_month'].dt.year
    monthly_data['month'] = monthly_data['year_month'].dt.month
    
    return monthly_data[['year', 'month', 'month_return']].dropna()


def calculate_rolling_cagr(data, years=4, crypto_data=None, fund_data=None):
    """
    Calculate rolling CAGR for a strategy
    
    Args:
        data: DataFrame with 'date', 'total_value', 'initial_value', 'start_date' columns
        years: Rolling period in years
        crypto_data: Optional DataFrame with crypto prices to include in output
        fund_data: Optional DataFrame with fund prices to include in output
    
    Returns:
        DataFrame with 'date', 'rolling_cagr', 'from_date', 'to_date', 'from_price', 'to_price' columns
    """
    if data.empty:
        return pd.DataFrame()
    
    data = data.sort_values('date').reset_index(drop=True)
    rolling_cagr_list = []
    
    days_window = int(years * 365.25)
    
    for i in range(len(data)):
        current_date = data.loc[i, 'date']
        current_value = data.loc[i, 'total_value']
        
        # Find the value from 'years' ago
        past_date = current_date - timedelta(days=days_window)
        past_data = data[data['date'] <= past_date]
        
        if not past_data.empty:
            past_value = past_data.iloc[-1]['total_value']
            from_date = past_data.iloc[-1]['date']
            actual_days = (current_date - from_date).days
            actual_years = actual_days / 365.25
            
            if actual_years > 0 and past_value > 0:
                cagr = (((current_value / past_value) ** (1 / actual_years)) - 1) * 100
                
                # Get crypto prices if available
                from_price = None
                to_price = None
                avg_dca_price = None
                
                if crypto_data is not None and 'price_crypto' in data.columns:
                    from_price = past_data.iloc[-1].get('price_crypto', None)
                    to_price = data.loc[i, 'price_crypto']
                    
                    # Calculate avg DCA price for 12-month window starting from from_date
                    monthly_prices = []
                    for month_idx in range(12):
                        purchase_date = from_date + timedelta(days=month_idx * 30)
                        price_data = crypto_data[crypto_data['date'] >= purchase_date]
                        if not price_data.empty:
                            monthly_prices.append(price_data.iloc[0]['price'])
                    
                    if monthly_prices:
                        # Harmonic mean for DCA (equal $ investment each month)
                        avg_dca_price = len(monthly_prices) / sum(1/p for p in monthly_prices)
                
                # Get fund prices if available (for Strategy 3)
                elif fund_data is not None:
                    # Match dates with fund_data
                    from_fund = fund_data[fund_data['date'] <= from_date]
                    to_fund = fund_data[fund_data['date'] <= current_date]
                    if not from_fund.empty and not to_fund.empty:
                        from_price = from_fund.iloc[-1]['price']
                        to_price = to_fund.iloc[-1]['price']
                
                rolling_cagr_list.append({
                    'date': current_date,
                    'rolling_cagr': cagr,
                    'from_date': from_date,
                    'to_date': current_date,
                    'from_price': from_price,
                    'to_price': to_price,
                    'avg_dca_price': avg_dca_price
                })
    
    return pd.DataFrame(rolling_cagr_list)

def get_top_n_dates_with_spacing(df, column, n=3, min_days_apart=30):
    """
    Get top N dates from dataframe with minimum spacing between them
    
    Args:
        df: DataFrame with 'start_date' and performance columns
        column: Column name to sort by (e.g., 'strategy_1_cagr')
        n: Number of top dates to return
        min_days_apart: Minimum days between selected dates
    
    Returns:
        DataFrame with top N dates that are at least min_days_apart from each other
    """
    # Sort by the specified column in descending order
    sorted_df = df.sort_values(column, ascending=False).reset_index(drop=True)
    
    selected = []
    for idx, row in sorted_df.iterrows():
        current_date = row['start_date']
        
        # Check if this date is far enough from already selected dates
        is_far_enough = True
        for selected_row in selected:
            days_diff = abs((current_date - selected_row['start_date']).days)
            if days_diff < min_days_apart:
                is_far_enough = False
                break
        
        if is_far_enough:
            selected.append(row)
            if len(selected) >= n:
                break
    
    return pd.DataFrame(selected)

def calculate_heatmap_data(crypto_data, fund_data, initial_amount, loan_pct, interest_rate, 
                          start_dates, end_date):
    """
    Calculate CAGR for multiple start dates to create heatmap
    
    Returns DataFrame with columns: start_date, strategy_1_cagr, strategy_2_cagr, strategy_3_cagr
    """
    results = []
    
    for start_date in start_dates:
        # Calculate Strategy 1 (Crypto-Backed VN Fund)
        s1_data = calculate_strategy_1(crypto_data, fund_data, initial_amount, loan_pct, 
                                      interest_rate, start_date, end_date)
        
        # Calculate Strategy 2 (DCA-Crypto-Backed VN Fund)
        s2_data = calculate_strategy_2_dca(crypto_data, fund_data, initial_amount, loan_pct,
                                          interest_rate, start_date, end_date)
        
        # Calculate Strategy 3 (Simple VN Fund)
        s3_data = calculate_strategy_3(fund_data, initial_amount, start_date, end_date)
        
        if s1_data.empty or s2_data.empty or s3_data.empty:
            continue
        
        # Get start and end values
        s1_start = s1_data.iloc[0]
        s1_final = s1_data.iloc[-1]
        s2_start = s2_data.iloc[0]
        s2_final = s2_data.iloc[-1]
        s3_start = s3_data.iloc[0]
        s3_final = s3_data.iloc[-1]
        
        # Calculate years elapsed
        days_elapsed = (s1_final['date'] - s1_start['date']).days
        years_elapsed = days_elapsed / 365.25
        
        if years_elapsed <= 0:
            continue
        
        # Calculate CAGR
        s1_cagr = (((s1_final['total_value'] / initial_amount) ** (1 / years_elapsed)) - 1) * 100
        s2_cagr = (((s2_final['total_value'] / initial_amount) ** (1 / years_elapsed)) - 1) * 100
        s3_cagr = (((s3_final['total_value'] / initial_amount) ** (1 / years_elapsed)) - 1) * 100
        
        # Get crypto prices at start and end
        crypto_start_price = s1_start.get('price_crypto', None)
        crypto_end_price = s1_final.get('price_crypto', None)
        
        results.append({
            'start_date': start_date,
            'strategy_1_cagr': s1_cagr,
            'strategy_2_cagr': s2_cagr,
            'strategy_3_cagr': s3_cagr,
            'strategy_1_value': s1_final['total_value'],
            'strategy_2_value': s2_final['total_value'],
            'strategy_3_value': s3_final['total_value'],
            'crypto_start_price': crypto_start_price,
            'crypto_end_price': crypto_end_price
        })
    
    return pd.DataFrame(results)

# Title
st.title("💰 Investment Strategy Comparison")
st.markdown("**Compare Crypto-Backed VN Fund vs Simple VN Fund investment**")

# Sidebar controls
st.sidebar.header("Strategy Parameters")

# Get available assets
crypto_assets, fund_assets = get_available_assets()

if not crypto_assets or not fund_assets:
    st.error("No crypto or fund data available. Please ensure data is imported.")
    st.stop()

# Get URL parameters
query_params = st.query_params

# Initial investment amount (hidden, using fixed value)
initial_amount = 10000000  # Fixed at $10M

# Loan percentage
loan_pct_default = int(query_params.get('loan_pct', 10))
loan_pct = st.sidebar.slider(
    "Loan Percentage (%)",
    min_value=1,
    max_value=50,
    value=loan_pct_default,
    step=1,
    help="Percentage of crypto value to borrow for VN fund investment"
)

# Interest rate
interest_rate_default = float(query_params.get('interest_rate', 6.0))
interest_rate = st.sidebar.slider(
    "Annual Interest Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=interest_rate_default,
    step=0.5,
    help="Annual interest rate on borrowed amount (compounds annually)"
)

# Cryptocurrency selection
crypto_default = query_params.get('crypto', 'BTC' if 'BTC' in crypto_assets else crypto_assets[0])
crypto_asset = st.sidebar.selectbox(
    "Cryptocurrency",
    options=crypto_assets,
    index=crypto_assets.index(crypto_default) if crypto_default in crypto_assets else 0,
    help="Select cryptocurrency for Crypto-Backed VN Fund strategy"
)

# VN Fund selection
fund_default = query_params.get('fund', 'DCDS' if 'DCDS' in fund_assets else fund_assets[0])
fund_asset = st.sidebar.selectbox(
    "VN Fund",
    options=fund_assets,
    index=fund_assets.index(fund_default) if fund_default in fund_assets else 0,
    help="Select VN fund for both strategies"
)

# Load price data
price_data = get_price_data([crypto_asset, fund_asset])

if price_data.empty:
    st.error("No price data available for selected assets.")
    st.stop()

# Separate crypto and fund data
crypto_data = price_data[price_data['asset_code'] == crypto_asset].copy()
fund_data = price_data[price_data['asset_code'] == fund_asset].copy()

# Get date range
min_date = max(crypto_data['date'].min(), fund_data['date'].min())
max_date = max(crypto_data['date'].max(), fund_data['date'].max())  # Use latest available date

# Date range selection
st.sidebar.subheader("Time Period")

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date_default = query_params.get('start_date', min_date.strftime('%Y-%m-%d'))
    start_date = st.date_input(
        "Start Date",
        value=pd.to_datetime(start_date_default).date() if start_date_default else min_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
        help="Date to start the investment strategy"
    )

with col2:
    end_date_default = query_params.get('end_date', max_date.strftime('%Y-%m-%d'))
    end_date = st.date_input(
        "End Date",
        value=pd.to_datetime(end_date_default).date() if end_date_default else max_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
        help="Date to end the comparison period"
    )

# Update URL parameters
st.query_params.update({
    'crypto': crypto_asset,
    'fund': fund_asset,
    'loan_pct': str(loan_pct),
    'interest_rate': str(interest_rate),
    'start_date': start_date.strftime('%Y-%m-%d'),
    'end_date': end_date.strftime('%Y-%m-%d')
})

# Convert to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Validate date range
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Calculate strategies
with st.spinner("Calculating strategies..."):
    strategy_1 = calculate_strategy_1(crypto_data, fund_data, initial_amount, loan_pct, 
                                     interest_rate, start_date, end_date)
    strategy_2 = calculate_strategy_2_dca(crypto_data, fund_data, initial_amount, loan_pct,
                                         interest_rate, start_date, end_date)
    strategy_3 = calculate_strategy_3(fund_data, initial_amount, start_date, end_date)

if strategy_1.empty or strategy_2.empty or strategy_3.empty:
    st.error("Unable to calculate strategies. Please check your date selection.")
    st.stop()


# DEBUG MODE - Skip to Debug Table section at the end
if DEBUG_MODE:
    st.warning("⚠️ DEBUG MODE ON - Showing only Debug Table (scroll to bottom)")
    st.stop()  # Stop here and skip all other sections
else:
    # Calculate rolling CAGR (pass crypto_data/fund_data for price information)
    rolling_cagr_4y_s1 = calculate_rolling_cagr(strategy_1, years=4, crypto_data=crypto_data)
    rolling_cagr_4y_s2 = calculate_rolling_cagr(strategy_2, years=4, crypto_data=crypto_data)
    rolling_cagr_4y_s3 = calculate_rolling_cagr(strategy_3, years=4, fund_data=fund_data)

    rolling_cagr_2y_s1 = calculate_rolling_cagr(strategy_1, years=2, crypto_data=crypto_data)
    rolling_cagr_2y_s2 = calculate_rolling_cagr(strategy_2, years=2, crypto_data=crypto_data)
    rolling_cagr_2y_s3 = calculate_rolling_cagr(strategy_3, years=2, fund_data=fund_data)

    rolling_cagr_1y_s1 = calculate_rolling_cagr(strategy_1, years=1, crypto_data=crypto_data)
    rolling_cagr_1y_s2 = calculate_rolling_cagr(strategy_2, years=1, crypto_data=crypto_data)
    rolling_cagr_1y_s3 = calculate_rolling_cagr(strategy_3, years=1, fund_data=fund_data)

# Skip charts section if in DEBUG_MODE
if not DEBUG_MODE:
    st.subheader("📈 Rolling CAGR Comparison")

    # 4-Year Rolling CAGR with Bitcoin Price
    st.markdown("**4-Year Rolling CAGR**")

    fig_4y = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.4],
    subplot_titles=('Rolling 4-Year CAGR', f'{crypto_asset} Price (USD)')
)

# Add Strategy 1 CAGR
if not rolling_cagr_4y_s1.empty:
    hover_text_s1 = []
    for _, row in rolling_cagr_4y_s1.iterrows():
        text = f"<b>Crypto-Backed VN Fund</b><br>From: {row['from_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('from_price')):
            text += f" ({crypto_asset}: ${row['from_price']:,.0f})"
        text += f"<br>To: {row['to_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('to_price')):
            text += f" ({crypto_asset}: ${row['to_price']:,.0f})"
        text += f"<br>CAGR: {row['rolling_cagr']:.2f}%"
        hover_text_s1.append(text)
    fig_4y.add_trace(go.Scatter(
        x=rolling_cagr_4y_s1['date'],
        y=rolling_cagr_4y_s1['rolling_cagr'],
        mode='lines',
        name='Crypto-Backed VN Fund',
        line=dict(color='#4ECDC4', width=2.5),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_s1
    ), row=1, col=1)

# Add Strategy 2 CAGR (DCA)
if not rolling_cagr_4y_s2.empty:
    hover_text_s2 = []
    for _, row in rolling_cagr_4y_s2.iterrows():
        text = f"<b>DCA-Crypto-Backed VN Fund</b><br>From: {row['from_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('from_price')):
            text += f" ({crypto_asset}: ${row['from_price']:,.0f})"
        text += f"<br>To: {row['to_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('to_price')):
            text += f" ({crypto_asset}: ${row['to_price']:,.0f})"
        # Add average BTC buy price for this specific rolling period
        avg_dca = row.get('avg_dca_price')
        if pd.notna(avg_dca) and avg_dca > 0:
            text += f"<br>Avg BTC Buy Price: ${avg_dca:,.0f}"
        text += f"<br>CAGR: {row['rolling_cagr']:.2f}%"
        hover_text_s2.append(text)
    fig_4y.add_trace(go.Scatter(
        x=rolling_cagr_4y_s2['date'],
        y=rolling_cagr_4y_s2['rolling_cagr'],
        mode='lines',
        name='DCA-Crypto-Backed VN Fund',
        line=dict(color='#9B59B6', width=2.5),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_s2
    ), row=1, col=1)

# Add Strategy 3 CAGR (Simple VN Fund)
if not rolling_cagr_4y_s3.empty:
    hover_text_s3 = []
    for _, row in rolling_cagr_4y_s3.iterrows():
        text = f"<b>Simple VN Fund</b><br>From: {row['from_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('from_price')):
            text += f" (NAV: {row['from_price']:,.0f} VND)"
        text += f"<br>To: {row['to_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('to_price')):
            text += f" (NAV: {row['to_price']:,.0f} VND)"
        text += f"<br>CAGR: {row['rolling_cagr']:.2f}%"
        hover_text_s3.append(text)
    fig_4y.add_trace(go.Scatter(
        x=rolling_cagr_4y_s3['date'],
        y=rolling_cagr_4y_s3['rolling_cagr'],
        mode='lines',
        name='Simple VN Fund',
        line=dict(color='#FF6B6B', width=2.5),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_s3
    ), row=1, col=1)

# Add 0% reference line
fig_4y.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

# Add Bitcoin price
fig_4y.add_trace(go.Scatter(
    x=strategy_1['date'],
    y=strategy_1['price_crypto'],
    mode='lines',
    name=f'{crypto_asset} Price',
    line=dict(color='#FFA500', width=2),
    hovertemplate=f'<b>{crypto_asset}</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:,.0f}}<extra></extra>'
), row=2, col=1)

# Add liquidation price line
fig_4y.add_trace(go.Scatter(
    x=strategy_1['date'],
    y=strategy_1['liquidation_price'],
    mode='lines',
    name='Liquidation Price',
    line=dict(color='#FF0000', width=2, dash='dot'),
    hovertemplate=f'<b>Liquidation Price</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:,.0f}}<extra></extra>'
), row=2, col=1)

# Highlight liquidation zones (where BTC price < liquidation price)
liquidated = strategy_1[strategy_1['price_crypto'] < strategy_1['liquidation_price']]
if not liquidated.empty:
    # Add shaded area for liquidation zones
    for date in liquidated['date']:
        fig_4y.add_vrect(
            x0=date - timedelta(days=1),
            x1=date + timedelta(days=1),
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0,
            row=2, col=1
        )

fig_4y.update_xaxes(title_text="Date", row=2, col=1)
fig_4y.update_yaxes(title_text="CAGR (%)", row=1, col=1)
fig_4y.update_yaxes(title_text="Price (USD)", type="log", row=2, col=1)  # Log scale for BTC price

fig_4y.update_layout(
    height=700,
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

st.plotly_chart(fig_4y, use_container_width=True)

# 2-Year Rolling CAGR with Bitcoin Price
st.markdown("**2-Year Rolling CAGR**")

fig_2y = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.4],
    subplot_titles=('Rolling 2-Year CAGR', f'{crypto_asset} Price (USD)')
)

# Add Strategy 1 CAGR
if not rolling_cagr_2y_s1.empty:
    hover_text_s1 = []
    for _, row in rolling_cagr_2y_s1.iterrows():
        text = f"<b>Crypto-Backed VN Fund</b><br>From: {row['from_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('from_price')):
            text += f" ({crypto_asset}: ${row['from_price']:,.0f})"
        text += f"<br>To: {row['to_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('to_price')):
            text += f" ({crypto_asset}: ${row['to_price']:,.0f})"
        text += f"<br>CAGR: {row['rolling_cagr']:.2f}%"
        hover_text_s1.append(text)
    fig_2y.add_trace(go.Scatter(
        x=rolling_cagr_2y_s1['date'],
        y=rolling_cagr_2y_s1['rolling_cagr'],
        mode='lines',
        name='Crypto-Backed VN Fund',
        line=dict(color='#4ECDC4', width=2.5),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_s1
    ), row=1, col=1)

# Add Strategy 2 CAGR (DCA)
if not rolling_cagr_2y_s2.empty:
    hover_text_s2 = []
    for _, row in rolling_cagr_2y_s2.iterrows():
        text = f"<b>DCA-Crypto-Backed VN Fund</b><br>From: {row['from_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('from_price')):
            text += f" ({crypto_asset}: ${row['from_price']:,.0f})"
        text += f"<br>To: {row['to_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('to_price')):
            text += f" ({crypto_asset}: ${row['to_price']:,.0f})"
        # Add average BTC buy price for this specific rolling period
        avg_dca = row.get('avg_dca_price')
        if pd.notna(avg_dca) and avg_dca > 0:
            text += f"<br>Avg BTC Buy Price: ${avg_dca:,.0f}"
        text += f"<br>CAGR: {row['rolling_cagr']:.2f}%"
        hover_text_s2.append(text)
    fig_2y.add_trace(go.Scatter(
        x=rolling_cagr_2y_s2['date'],
        y=rolling_cagr_2y_s2['rolling_cagr'],
        mode='lines',
        name='DCA-Crypto-Backed VN Fund',
        line=dict(color='#9B59B6', width=2.5),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_s2
    ), row=1, col=1)

# Add Strategy 3 CAGR (Simple VN Fund)
if not rolling_cagr_2y_s3.empty:
    hover_text_s3 = []
    for _, row in rolling_cagr_2y_s3.iterrows():
        text = f"<b>Simple VN Fund</b><br>From: {row['from_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('from_price')):
            text += f" (NAV: {row['from_price']:,.0f} VND)"
        text += f"<br>To: {row['to_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('to_price')):
            text += f" (NAV: {row['to_price']:,.0f} VND)"
        text += f"<br>CAGR: {row['rolling_cagr']:.2f}%"
        hover_text_s3.append(text)
    fig_2y.add_trace(go.Scatter(
        x=rolling_cagr_2y_s3['date'],
        y=rolling_cagr_2y_s3['rolling_cagr'],
        mode='lines',
        name='Simple VN Fund',
        line=dict(color='#FF6B6B', width=2.5),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_s3
    ), row=1, col=1)

# Add 0% reference line
fig_2y.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

# Add Bitcoin price
fig_2y.add_trace(go.Scatter(
    x=strategy_1['date'],
    y=strategy_1['price_crypto'],
    mode='lines',
    name=f'{crypto_asset} Price',
    line=dict(color='#FFA500', width=2),
    hovertemplate=f'<b>{crypto_asset}</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:,.0f}}<extra></extra>'
), row=2, col=1)

fig_2y.update_xaxes(title_text="Date", row=2, col=1)
fig_2y.update_yaxes(title_text="CAGR (%)", row=1, col=1)
fig_2y.update_yaxes(title_text="Price (USD)", type="log", row=2, col=1)  # Log scale for BTC price

fig_2y.update_layout(
    height=700,
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

st.plotly_chart(fig_2y, use_container_width=True)

# 1-Year Rolling CAGR with Bitcoin Price
st.markdown("**1-Year Rolling CAGR**")

fig_1y = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.4],
    subplot_titles=('Rolling 1-Year CAGR', f'{crypto_asset} Price (USD)')
)

# Add Strategy 1 CAGR
if not rolling_cagr_1y_s1.empty:
    hover_text_s1 = []
    for _, row in rolling_cagr_1y_s1.iterrows():
        text = f"<b>Crypto-Backed VN Fund</b><br>From: {row['from_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('from_price')):
            text += f" ({crypto_asset}: ${row['from_price']:,.0f})"
        text += f"<br>To: {row['to_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('to_price')):
            text += f" ({crypto_asset}: ${row['to_price']:,.0f})"
        text += f"<br>CAGR: {row['rolling_cagr']:.2f}%"
        hover_text_s1.append(text)
    fig_1y.add_trace(go.Scatter(
        x=rolling_cagr_1y_s1['date'],
        y=rolling_cagr_1y_s1['rolling_cagr'],
        mode='lines',
        name='Crypto-Backed VN Fund',
        line=dict(color='#4ECDC4', width=2.5),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_s1
    ), row=1, col=1)

# Add Strategy 2 CAGR (DCA)
if not rolling_cagr_1y_s2.empty:
    hover_text_s2 = []
    for _, row in rolling_cagr_1y_s2.iterrows():
        text = f"<b>DCA-Crypto-Backed VN Fund</b><br>From: {row['from_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('from_price')):
            text += f" ({crypto_asset}: ${row['from_price']:,.0f})"
        text += f"<br>To: {row['to_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('to_price')):
            text += f" ({crypto_asset}: ${row['to_price']:,.0f})"
        # Add average BTC buy price for this specific rolling period
        avg_dca = row.get('avg_dca_price')
        if pd.notna(avg_dca) and avg_dca > 0:
            text += f"<br>Avg BTC Buy Price: ${avg_dca:,.0f}"
        text += f"<br>CAGR: {row['rolling_cagr']:.2f}%"
        hover_text_s2.append(text)
    fig_1y.add_trace(go.Scatter(
        x=rolling_cagr_1y_s2['date'],
        y=rolling_cagr_1y_s2['rolling_cagr'],
        mode='lines',
        name='DCA-Crypto-Backed VN Fund',
        line=dict(color='#9B59B6', width=2.5),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_s2
    ), row=1, col=1)

# Add Strategy 3 CAGR (Simple VN Fund)
if not rolling_cagr_1y_s3.empty:
    hover_text_s3 = []
    for _, row in rolling_cagr_1y_s3.iterrows():
        text = f"<b>Simple VN Fund</b><br>From: {row['from_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('from_price')):
            text += f" (NAV: {row['from_price']:,.0f} VND)"
        text += f"<br>To: {row['to_date'].strftime('%d/%m/%Y')}"
        if pd.notna(row.get('to_price')):
            text += f" (NAV: {row['to_price']:,.0f} VND)"
        text += f"<br>CAGR: {row['rolling_cagr']:.2f}%"
        hover_text_s3.append(text)
    fig_1y.add_trace(go.Scatter(
        x=rolling_cagr_1y_s3['date'],
        y=rolling_cagr_1y_s3['rolling_cagr'],
        mode='lines',
        name='Simple VN Fund',
        line=dict(color='#FF6B6B', width=2.5),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text_s3
    ), row=1, col=1)

# Add 0% reference line
fig_1y.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

# Add Bitcoin price
fig_1y.add_trace(go.Scatter(
    x=strategy_1['date'],
    y=strategy_1['price_crypto'],
    mode='lines',
    name=f'{crypto_asset} Price',
    line=dict(color='#FFA500', width=2),
    hovertemplate=f'<b>{crypto_asset}</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:,.0f}}<extra></extra>'
), row=2, col=1)

fig_1y.update_xaxes(title_text="Date", row=2, col=1)
fig_1y.update_yaxes(title_text="CAGR (%)", row=1, col=1)
fig_1y.update_yaxes(title_text="Price (USD)", type="log", row=2, col=1)  # Log scale for BTC price

fig_1y.update_layout(
    height=700,
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

st.plotly_chart(fig_1y, use_container_width=True)

# Summary metrics (removed as per user request)
# st.subheader("📊 Performance Summary")

# Heatmap section (Performance Summary removed as per user request)
st.markdown("---")
st.subheader("🔥 CAGR Heatmap: Different Start Dates")
st.markdown("*See how CAGR varies based on when you started the investment*")

# Generate start dates (monthly intervals) - use selected start_date and end_date
heatmap_start = start_date  # Use selected start_date instead of min_date
heatmap_end_for_starts = end_date - timedelta(days=365)  # At least 1 year of data before end_date

if heatmap_start < heatmap_end_for_starts:
    # Generate monthly start dates
    start_dates = pd.date_range(start=heatmap_start, end=heatmap_end_for_starts, freq='MS').tolist()
    
    with st.spinner("Calculating heatmap data..."):
        heatmap_data = calculate_heatmap_data(
            crypto_data, fund_data, initial_amount, loan_pct, interest_rate,
            start_dates, end_date  # Use selected end_date instead of max_date
        )
    
    if not heatmap_data.empty:
        # Prepare data for heatmap
        heatmap_df = heatmap_data.copy()
        heatmap_df['start_date_str'] = heatmap_df['start_date'].dt.strftime('%Y-%m')
        
        # Create heatmap data in wide format
        heatmap_matrix = pd.DataFrame({
            'Start Date': heatmap_df['start_date_str'],
            'Crypto-Backed VN Fund': heatmap_df['strategy_1_cagr'],
            'DCA-Crypto-Backed VN Fund': heatmap_df['strategy_2_cagr'],
            'Simple VN Fund': heatmap_df['strategy_3_cagr']
        })
        
        # Transpose for better visualization
        heatmap_matrix_t = heatmap_matrix.set_index('Start Date').T
        
        # Prepare customdata for tooltip with crypto prices
        # Create a 2D array matching the heatmap dimensions
        customdata_array = []
        for strategy_idx in range(len(heatmap_matrix_t.index)):
            row_data = []
            for date_idx in range(len(heatmap_df)):
                row_data.append([
                    heatmap_df.iloc[date_idx].get('crypto_start_price', 0),
                    heatmap_df.iloc[date_idx].get('crypto_end_price', 0)
                ])
            customdata_array.append(row_data)
        
        # Create heatmap (without text overlay)
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_matrix_t.values,
            x=heatmap_matrix_t.columns,
            y=heatmap_matrix_t.index,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(
                title="CAGR (%)",
                ticksuffix="%"
            ),
            customdata=customdata_array,
            hovertemplate='<b>%{y}</b><br>' +
                         'Start: %{x}<br>' +
                         'CAGR: %{z:.2f}%<br>' +
                         f'{crypto_asset} Start: $' + '%{customdata[0]:,.0f}<br>' +
                         f'{crypto_asset} End: $' + '%{customdata[1]:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title=f"CAGR % by Start Date and Strategy (End Date: {end_date.strftime('%Y-%m-%d')})",
            xaxis_title="Start Date (Year-Month)",
            yaxis_title="Strategy",
            height=300,
            xaxis=dict(tickangle=-45),
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Heatmap insights
        st.markdown("**📖 Heatmap Insights:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Top 3 start dates for Strategy 1 (with minimum 3 months spacing)
            top3_s1 = get_top_n_dates_with_spacing(heatmap_df, 'strategy_1_cagr', n=3, min_days_apart=90)
            st.markdown(f"**Top 3 Start Dates (Crypto-Backed VN Fund):**")
            medals = ['🥇', '🥈', '🥉']
            for i, (idx, row) in enumerate(top3_s1.iterrows()):
                medal = medals[i] if i < len(medals) else '•'
                st.markdown(
                    f"{medal} **{row['start_date'].strftime('%d/%m/%Y')}**: "
                    f"S1: {row['strategy_1_cagr']:.2f}% | S2: {row['strategy_2_cagr']:.2f}% | S3: {row['strategy_3_cagr']:.2f}%"
                )
            
        with col2:
            # Top 3 start dates for Strategy 2 (DCA) (with minimum 3 months spacing)
            top3_s2 = get_top_n_dates_with_spacing(heatmap_df, 'strategy_2_cagr', n=3, min_days_apart=90)
            st.markdown(f"**Top 3 Start Dates (DCA-Crypto-Backed VN Fund):**")
            medals = ['🥇', '🥈', '🥉']
            for i, (idx, row) in enumerate(top3_s2.iterrows()):
                medal = medals[i] if i < len(medals) else '•'
                st.markdown(
                    f"{medal} **{row['start_date'].strftime('%d/%m/%Y')}**: "
                    f"S1: {row['strategy_1_cagr']:.2f}% | S2: {row['strategy_2_cagr']:.2f}% | S3: {row['strategy_3_cagr']:.2f}%"
                )
        
        with col3:
            # Top 3 start dates for Strategy 3 (Simple VN Fund) (with minimum 3 months spacing)
            top3_s3 = get_top_n_dates_with_spacing(heatmap_df, 'strategy_3_cagr', n=3, min_days_apart=90)
            st.markdown(f"**Top 3 Start Dates (Simple VN Fund):**")
            medals = ['🥇', '🥈', '🥉']
            for i, (idx, row) in enumerate(top3_s3.iterrows()):
                medal = medals[i] if i < len(medals) else '•'
                st.markdown(
                    f"{medal} **{row['start_date'].strftime('%d/%m/%Y')}**: "
                    f"S1: {row['strategy_1_cagr']:.2f}% | S2: {row['strategy_2_cagr']:.2f}% | S3: {row['strategy_3_cagr']:.2f}%"
                )
        
        # Strategy comparison across all start dates
        st.markdown("**Overall Strategy Performance:**")
        
        # Calculate wins for each strategy
        s1_wins = ((heatmap_df['strategy_1_cagr'] > heatmap_df['strategy_2_cagr']) & 
                   (heatmap_df['strategy_1_cagr'] > heatmap_df['strategy_3_cagr'])).sum()
        s2_wins = ((heatmap_df['strategy_2_cagr'] > heatmap_df['strategy_1_cagr']) & 
                   (heatmap_df['strategy_2_cagr'] > heatmap_df['strategy_3_cagr'])).sum()
        s3_wins = ((heatmap_df['strategy_3_cagr'] > heatmap_df['strategy_1_cagr']) & 
                   (heatmap_df['strategy_3_cagr'] > heatmap_df['strategy_2_cagr'])).sum()
        total = len(heatmap_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Crypto-Backed Wins", f"{s1_wins}/{total}", f"{(s1_wins/total)*100:.1f}%")
        with col2:
            st.metric("DCA-Crypto-Backed Wins", f"{s2_wins}/{total}", f"{(s2_wins/total)*100:.1f}%")
        with col3:
            st.metric("Simple Fund Wins", f"{s3_wins}/{total}", f"{(s3_wins/total)*100:.1f}%")
    else:
        st.info("Not enough data to generate heatmap.")
else:
    st.info("Not enough historical data to generate heatmap. Need at least 1 year of data.")


# Yearly Drawdown Analysis
st.markdown("---")
st.subheader("📉 Yearly Maximum Drawdown Analysis")
st.markdown("Maximum drawdown shows the largest peak-to-trough decline within each year. Lower (less negative) is better.")

# Calculate yearly drawdowns for all strategies
drawdown_s1 = calculate_yearly_drawdown(strategy_1)
drawdown_s2 = calculate_yearly_drawdown(strategy_2)
drawdown_s3 = calculate_yearly_drawdown(strategy_3)

if not drawdown_s1.empty and not drawdown_s2.empty and not drawdown_s3.empty:
    # Merge all drawdowns
    drawdown_s1['strategy'] = 'Crypto-Backed VN Fund'
    drawdown_s2['strategy'] = 'DCA-Crypto-Backed VN Fund'
    drawdown_s3['strategy'] = 'Simple VN Fund'
    
    all_drawdowns = pd.concat([drawdown_s1, drawdown_s2, drawdown_s3])
    
    # Create line chart
    fig_drawdown = go.Figure()
    
    fig_drawdown.add_trace(go.Scatter(
        x=drawdown_s1['year'],
        y=drawdown_s1['max_drawdown'],
        mode='lines+markers',
        name='Crypto-Backed VN Fund',
        line=dict(color='#4ECDC4', width=2.5),
        marker=dict(size=8)
    ))
    
    fig_drawdown.add_trace(go.Scatter(
        x=drawdown_s2['year'],
        y=drawdown_s2['max_drawdown'],
        mode='lines+markers',
        name='DCA-Crypto-Backed VN Fund',
        line=dict(color='#9B59B6', width=2.5),
        marker=dict(size=8)
    ))
    
    fig_drawdown.add_trace(go.Scatter(
        x=drawdown_s3['year'],
        y=drawdown_s3['max_drawdown'],
        mode='lines+markers',
        name='Simple VN Fund',
        line=dict(color='#FF6B6B', width=2.5),
        marker=dict(size=8)
    ))
    
    fig_drawdown.update_layout(
        title="Yearly Maximum Drawdown Comparison",
        xaxis_title="Year",
        yaxis_title="Max Drawdown (%)",
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_drawdown, use_container_width=True)
    
    # Create comparison table
    st.markdown("**📊 Yearly Drawdown Table:**")
    
    # Pivot data for table
    pivot_data = []
    years = sorted(all_drawdowns['year'].unique())
    
    for year in years:
        year_data = {'Year': int(year)}
        s1_dd = drawdown_s1[drawdown_s1['year'] == year]['max_drawdown'].values
        s2_dd = drawdown_s2[drawdown_s2['year'] == year]['max_drawdown'].values
        s3_dd = drawdown_s3[drawdown_s3['year'] == year]['max_drawdown'].values
        
        year_data['Crypto-Backed'] = f"{s1_dd[0]:.2f}%" if len(s1_dd) > 0 else "N/A"
        year_data['DCA-Crypto-Backed'] = f"{s2_dd[0]:.2f}%" if len(s2_dd) > 0 else "N/A"
        year_data['Simple Fund'] = f"{s3_dd[0]:.2f}%" if len(s3_dd) > 0 else "N/A"
        
        # Find best (least negative) drawdown
        dds = []
        if len(s1_dd) > 0: dds.append(('Crypto-Backed', s1_dd[0]))
        if len(s2_dd) > 0: dds.append(('DCA-Crypto-Backed', s2_dd[0]))
        if len(s3_dd) > 0: dds.append(('Simple Fund', s3_dd[0]))
        
        if dds:
            best_strategy = max(dds, key=lambda x: x[1])[0]  # Max because drawdowns are negative
            year_data['Best Strategy'] = best_strategy
        else:
            year_data['Best Strategy'] = "N/A"
        
        pivot_data.append(year_data)
    
    df_table = pd.DataFrame(pivot_data)
    
    # Style the table
    def highlight_best(row):
        styles = [''] * len(row)
        best = row['Best Strategy']
        if best == 'Crypto-Backed':
            styles[1] = 'background-color: #90EE90'
        elif best == 'DCA-Crypto-Backed':
            styles[2] = 'background-color: #90EE90'
        elif best == 'Simple Fund':
            styles[3] = 'background-color: #90EE90'
        return styles
    
    st.dataframe(
        df_table.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Not enough data to calculate yearly drawdowns.")



# ========== MONTHLY PORTFOLIO DEBUG TABLE (Final Section) ==========
st.markdown("---")
st.subheader("🔍 Monthly Portfolio Debug Table")
    

debug_start_date_main = st.date_input(
    "Debug Start Date",
    value=start_date,
    min_value=min_date.date(),
    max_value=end_date.date() if hasattr(end_date, 'date') else end_date,
    key="debug_start_date_main"
)

# Convert debug_start_date to datetime
debug_start_main = pd.to_datetime(debug_start_date_main)

# Calculate number of months to display (from debug_start to end_date)
debug_end_main = end_date
months_diff_main = (debug_end_main.year - debug_start_main.year) * 12 + (debug_end_main.month - debug_start_main.month)
display_months_main = max(1, months_diff_main + 1)

# Get the ACTUAL first BTC price and date from debug_start
first_btc_data_main = crypto_data[crypto_data['date'] >= debug_start_main]

if not first_btc_data_main.empty:
        actual_first_date_main = first_btc_data_main.iloc[0]['date']
        initial_btc_price_main = first_btc_data_main.iloc[0]['price']
        dca_installments_main = 12
        monthly_investment_main = initial_amount / dca_installments_main
        
        # S1 calculations
        s1_btc_holdings_main = initial_amount / initial_btc_price_main
        s1_loan_amount_main = initial_amount * (loan_pct / 100)
        
        # Calculate S2 DCA purchases
        s2_purchase_records_main = []
        cumulative_btc_main = 0
        cumulative_loan_main = 0
        cumulative_fund_units_main = 0
        
        for month_idx in range(dca_installments_main):
            purchase_date = actual_first_date_main + timedelta(days=month_idx * 30)
            crypto_at_purchase = crypto_data[crypto_data['date'] >= purchase_date]
            if crypto_at_purchase.empty:
                break
            
            btc_price_at_purchase = crypto_at_purchase.iloc[0]['price']
            actual_purchase_date = crypto_at_purchase.iloc[0]['date']
            btc_purchased = monthly_investment_main / btc_price_at_purchase
            cumulative_btc_main += btc_purchased
            loan_this_month = monthly_investment_main * (loan_pct / 100)
            cumulative_loan_main += loan_this_month
            
            fund_at_purchase = fund_data[fund_data['date'] >= actual_purchase_date]
            if not fund_at_purchase.empty:
                fund_price = fund_at_purchase.iloc[0]['price']
                fund_price_usd = fund_price / VND_TO_USD
                fund_units_purchased = loan_this_month / fund_price_usd
                cumulative_fund_units_main += fund_units_purchased
            
            s2_purchase_records_main.append({
                'month_idx': month_idx + 1,
                'date': actual_purchase_date,
                'btc_price': btc_price_at_purchase,
                'btc_purchased': btc_purchased,
                'cumulative_btc': cumulative_btc_main,
                'loan_amount': loan_this_month,
                'cumulative_loan': cumulative_loan_main,
                'cumulative_fund_units': cumulative_fund_units_main
            })
        
        # Build debug table data
        debug_data_main = []
        first_fund_data_main = fund_data[fund_data['date'] >= actual_first_date_main]
        initial_fund_price_main = first_fund_data_main.iloc[0]['price'] if not first_fund_data_main.empty else 0
        
        # Add initial row
        initial_row_main = {
            'Date': actual_first_date_main.strftime('%Y-%m-%d') + ' (Start)',
            'BTC Price (End)': f"${initial_btc_price_main:,.0f}",
            'Fund NAV (VND)': f"{initial_fund_price_main:,.0f}",
            'S1: BTC Holdings': f"{s1_btc_holdings_main:.4f}",
            'S1: Avg BTC Price': f"${initial_btc_price_main:,.0f}",
            'S1: Crypto Value': f"${initial_amount:,.0f}",
            'S1: Fund Value': f"$0",
            'S1: Debt': f"$0",
            'S1: Valuation': f"${initial_amount:,.0f}",
            'S1: CAGR': "0.0%",
            'S2: BTC Holdings': "0.0000",
            'S2: Avg BTC Price': "$0",
            'S2: Crypto Value': f"$0",
            'S2: Fund Value': f"$0",
            'S2: Cash': f"${initial_amount:,.0f}",
            'S2: Debt': f"$0",
            'S2: Valuation': f"${initial_amount:,.0f}",
            'S2: CAGR': "0.0%",
            'S3: Valuation': f"${initial_amount:,.0f}",
            'S3: CAGR': "0.0%",
            '_is_initial': True
        }
        debug_data_main.append(initial_row_main)
        
        for month_idx in range(display_months_main):
            target_date = actual_first_date_main + pd.DateOffset(months=month_idx)
            month_end = target_date + pd.offsets.MonthEnd(0)
            
            btc_at_end = crypto_data[crypto_data['date'] <= month_end]
            if btc_at_end.empty:
                continue
            btc_price_end = btc_at_end.iloc[-1]['price']
            actual_month_end = btc_at_end.iloc[-1]['date']
            
            fund_at_end = fund_data[fund_data['date'] <= month_end]
            fund_nav_vnd = fund_at_end.iloc[-1]['price'] if not fund_at_end.empty else 0
            
            # S1 calculations
            s1_crypto_value = s1_btc_holdings_main * btc_price_end
            if not first_fund_data_main.empty:
                s1_fund_units = s1_loan_amount_main / (initial_fund_price_main / VND_TO_USD)
                s1_fund_value = s1_fund_units * (fund_nav_vnd / VND_TO_USD)
            else:
                s1_fund_value = 0
            
            years_elapsed = (actual_month_end - actual_first_date_main).days / 365.25
            s1_debt = s1_loan_amount_main * ((1 + interest_rate / 100) ** years_elapsed)
            s1_valuation = s1_crypto_value + s1_fund_value - s1_debt
            
            # S2 calculations
            dca_months_completed = min(month_idx + 1, len(s2_purchase_records_main))
            
            if dca_months_completed > 0 and dca_months_completed <= len(s2_purchase_records_main):
                s2_record = s2_purchase_records_main[dca_months_completed - 1]
                s2_btc_holdings = s2_record['cumulative_btc']
                s2_cumulative_fund_units = s2_record['cumulative_fund_units']
                s2_cash = initial_amount * max(0, (dca_installments_main - dca_months_completed)) / dca_installments_main
                s2_crypto_value = s2_btc_holdings * btc_price_end
                s2_fund_value = s2_cumulative_fund_units * (fund_nav_vnd / VND_TO_USD)
                
                s2_debt = 0
                for i in range(dca_months_completed):
                    rec = s2_purchase_records_main[i]
                    years = (actual_month_end - rec['date']).days / 365.25
                    if years >= 0:
                        s2_debt += rec['loan_amount'] * ((1 + interest_rate / 100) ** years)
                
                s2_valuation = s2_crypto_value + s2_fund_value - s2_debt + s2_cash
                total_btc_cost = sum([s2_purchase_records_main[i]['btc_price'] * s2_purchase_records_main[i]['btc_purchased'] 
                                     for i in range(dca_months_completed)])
                s2_avg_btc_price = total_btc_cost / s2_btc_holdings if s2_btc_holdings > 0 else 0
            else:
                s2_btc_holdings = 0
                s2_crypto_value = 0
                s2_fund_value = 0
                s2_debt = 0
                s2_cash = initial_amount
                s2_valuation = s2_cash
                s2_avg_btc_price = 0
            
            # S3 calculations
            if not first_fund_data_main.empty:
                s3_fund_units = initial_amount / (initial_fund_price_main / VND_TO_USD)
                s3_fund_value = s3_fund_units * (fund_nav_vnd / VND_TO_USD)
                s3_valuation = s3_fund_value
            else:
                s3_fund_value = 0
                s3_valuation = 0
            
            # Calculate CAGR
            years_for_cagr = (actual_month_end - actual_first_date_main).days / 365.25
            if years_for_cagr > 0:
                s1_cagr = (((s1_valuation / initial_amount) ** (1 / years_for_cagr)) - 1) * 100
                s2_cagr = (((s2_valuation / initial_amount) ** (1 / years_for_cagr)) - 1) * 100
                s3_cagr = (((s3_valuation / initial_amount) ** (1 / years_for_cagr)) - 1) * 100
            else:
                s1_cagr = 0
                s2_cagr = 0
                s3_cagr = 0
            
            debug_row_main = {
                'Date': actual_month_end.strftime('%Y-%m'),
                'BTC Price (End)': f"${btc_price_end:,.0f}",
                'Fund NAV (VND)': f"{fund_nav_vnd:,.0f}",
                'S1: BTC Holdings': f"{s1_btc_holdings_main:.4f}",
                'S1: Avg BTC Price': f"${initial_btc_price_main:,.0f}",
                'S1: Crypto Value': f"${s1_crypto_value:,.0f}",
                'S1: Fund Value': f"${s1_fund_value:,.0f}",
                'S1: Debt': f"${s1_debt:,.0f}",
                'S1: Valuation': f"${s1_valuation:,.0f}",
                'S1: CAGR': f"{s1_cagr:.1f}%",
                'S2: BTC Holdings': f"{s2_btc_holdings:.4f}",
                'S2: Avg BTC Price': f"${s2_avg_btc_price:,.0f}",
                'S2: Crypto Value': f"${s2_crypto_value:,.0f}",
                'S2: Fund Value': f"${s2_fund_value:,.0f}",
                'S2: Cash': f"${s2_cash:,.0f}",
                'S2: Debt': f"${s2_debt:,.0f}",
                'S2: Valuation': f"${s2_valuation:,.0f}",
                'S2: CAGR': f"{s2_cagr:.1f}%",
                'S3: Valuation': f"${s3_valuation:,.0f}",
                'S3: CAGR': f"{s3_cagr:.1f}%",
                '_is_initial': False
            }
            debug_data_main.append(debug_row_main)
        
        if debug_data_main:
            df_debug_main = pd.DataFrame(debug_data_main)
            
            # Show first 30 and last 30 rows
            total_rows = len(df_debug_main)
            if total_rows > 60:
                df_first_30 = df_debug_main.head(30)
                df_last_30 = df_debug_main.tail(30)
                separator = {col: '...' for col in df_debug_main.columns if col != '_is_initial'}
                separator['_is_initial'] = False
                df_separator = pd.DataFrame([separator])
                df_display_main = pd.concat([df_first_30, df_separator, df_last_30], ignore_index=True)
            else:
                df_display_main = df_debug_main
            
            is_initial_flags_main = df_display_main['_is_initial'].copy()
            df_display_main = df_display_main.drop(columns=['_is_initial'])
            
            def highlight_columns_main(s):
                styles = []
                for idx in range(len(s)):
                    if idx < len(is_initial_flags_main) and is_initial_flags_main.iloc[idx]:
                        styles.append('background-color: rgba(255, 255, 0, 0.2)')
                    elif 'Valuation' in s.name:
                        styles.append('background-color: rgba(61, 157, 243, 0.2)')
                    elif 'CAGR' in s.name:
                        styles.append('background-color: rgba(34, 139, 34, 0.3)')
                    else:
                        styles.append('')
                return styles
            
            column_config_main = {col: st.column_config.TextColumn(col, disabled=True) 
                                 for col in df_display_main.columns}
            
            st.dataframe(
                df_display_main.style.apply(highlight_columns_main), 
                use_container_width=True, 
                hide_index=True,
                column_config=column_config_main
            )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>
    <p>💡 <b>Note:</b> Strategy 1 borrows against crypto holdings to invest in VN funds. Interest accumulates as debt.</p>
    <p>Exchange rate: 1 USD = 26,000 VND (fixed)</p>
    <p>⚠️ Past performance does not guarantee future results. This is for educational purposes only.</p>
</div>
""", unsafe_allow_html=True)
