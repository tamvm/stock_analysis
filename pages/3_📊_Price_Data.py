import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import DataProcessor
from metrics_calculator import MetricsCalculator

# Page config
st.set_page_config(
    page_title="Price Data Listing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define colors
CHART_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'
]

# Initialize processors
@st.cache_resource
def init_processors():
    return DataProcessor(), MetricsCalculator()

processor, calc = init_processors()

# Title
st.title("📊 Price Data Listing")
st.markdown("View and filter historical price data for all assets")

# Load available assets
try:
    assets_df = processor.get_asset_list()
    fund_assets = sorted(assets_df[assets_df['asset_type'] == 'vn_fund']['asset_code'].tolist())
    etf_assets = sorted(assets_df[assets_df['asset_type'] == 'us_etf']['asset_code'].tolist())
    benchmark_assets = sorted(assets_df[assets_df['asset_type'].isin(['benchmark', 'vn_index'])]['asset_code'].tolist())
    stock_assets = sorted(assets_df[assets_df['asset_type'] == 'us_stock']['asset_code'].tolist())
    crypto_assets = sorted(assets_df[assets_df['asset_type'] == 'crypto']['asset_code'].tolist())
    commodity_assets = sorted(assets_df[assets_df['asset_type'] == 'commodity']['asset_code'].tolist())

    # Create organized list with clear labels
    organized_options = []
    if fund_assets:
        organized_options.extend([f"📈 {asset} (Fund)" for asset in fund_assets])
    if etf_assets:
        organized_options.extend([f"🏛️ {asset} (ETF)" for asset in etf_assets])
    if stock_assets:
        organized_options.extend([f"📊 {asset} (Stock)" for asset in stock_assets])
    if crypto_assets:
        organized_options.extend([f"₿ {asset} (Crypto)" for asset in crypto_assets])
    if commodity_assets:
        organized_options.extend([f"🥇 {asset} (Commodity)" for asset in commodity_assets])
    if benchmark_assets:
        organized_options.extend([f"📉 {asset} (Benchmark)" for asset in benchmark_assets])

    # Create mapping from display names to asset codes
    display_to_asset = {}
    for asset in fund_assets:
        display_name = f"📈 {asset} (Fund)"
        display_to_asset[display_name] = asset
    for asset in etf_assets:
        display_name = f"🏛️ {asset} (ETF)"
        display_to_asset[display_name] = asset
    for asset in stock_assets:
        display_name = f"📊 {asset} (Stock)"
        display_to_asset[display_name] = asset
    for asset in crypto_assets:
        display_name = f"₿ {asset} (Crypto)"
        display_to_asset[display_name] = asset
    for asset in commodity_assets:
        display_name = f"🥇 {asset} (Commodity)"
        display_to_asset[display_name] = asset
    for asset in benchmark_assets:
        display_name = f"📉 {asset} (Benchmark)"
        display_to_asset[display_name] = asset

    all_assets = fund_assets + etf_assets + stock_assets + crypto_assets + commodity_assets + benchmark_assets
except Exception as e:
    st.error("Database not found. Please run data_processor.py first to load data.")
    st.stop()

# Sidebar filters
st.sidebar.header("🎛️ Filter Settings")

# Back to dashboard button
if st.sidebar.button("🏠 Back to Dashboard", help="Return to main dashboard"):
    st.switch_page("dashboard.py")

# Asset selection (removed asset type filter)
st.sidebar.subheader("Asset Selection")

# Determine default value - only set on initial load
if 'selected_display_assets' in st.session_state and st.session_state.selected_display_assets is not None:
    # Widget already has a value, use it (prevents reset on rerun)
    default_display_assets = st.session_state.selected_display_assets
else:
    # Initial default: first 5 assets
    default_display_assets = organized_options[:5] if len(organized_options) >= 5 else organized_options

# Callback function to handle asset selection changes
def on_display_assets_change():
    """Update session state when assets are selected"""
    # No additional processing needed, just let the widget update
    pass

selected_display_assets = st.sidebar.multiselect(
    "📋 **Available Assets:**",
    options=organized_options,
    default=default_display_assets,
    help="📈 Funds | 🏛️ ETFs | 📊 Benchmarks",
    key="selected_display_assets",
    on_change=on_display_assets_change
)

# Convert back to asset codes for processing
selected_assets = [display_to_asset[display_name] for display_name in selected_display_assets]

if not selected_assets:
    st.warning("Please select at least one asset to analyze.")
    st.stop()

# Get data range from selected assets
@st.cache_data
def get_asset_date_range(asset_codes):
    try:
        price_data = processor.get_price_data(asset_codes)
        if not price_data.empty:
            min_date = price_data['date'].min().date()
            max_date = price_data['date'].max().date()
            return min_date, max_date
        return None, None
    except:
        return None, None

min_date, max_date = get_asset_date_range(selected_assets)

if min_date is None or max_date is None:
    st.error("Unable to load date range for selected assets.")
    st.stop()

# Date range filter
st.sidebar.subheader("📅 Date Range Filter")

# Quick date range buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("📅 Max Range", help="Set to full data range available"):
        st.session_state.filter_start_date = min_date
        st.session_state.filter_end_date = max_date
        st.rerun()

with col2:
    if st.button("📅 Last 1Y", help="Set to last 1 year"):
        end_date_1y = max_date
        start_date_1y = max(min_date, end_date_1y - timedelta(days=365))
        st.session_state.filter_start_date = start_date_1y
        st.session_state.filter_end_date = end_date_1y
        st.rerun()

# Date input widgets
col1, col2 = st.sidebar.columns(2)
with col1:
    # Ensure the default value is within valid range
    default_start_date = st.session_state.get('filter_start_date', max(min_date, max_date - timedelta(days=365)))
    if default_start_date > max_date:
        default_start_date = max_date
    elif default_start_date < min_date:
        default_start_date = min_date

    start_date = st.date_input(
        "From:",
        value=default_start_date,
        min_value=min_date,
        max_value=max_date,
        key="filter_start_date"
    )
with col2:
    # Ensure the default value is within valid range
    default_end_date = st.session_state.get('filter_end_date', max_date)
    if default_end_date > max_date:
        default_end_date = max_date
    elif default_end_date < min_date:
        default_end_date = min_date

    end_date = st.date_input(
        "To:",
        value=default_end_date,
        min_value=min_date,
        max_value=max_date,
        key="filter_end_date"
    )

# Validate date range
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Analysis settings
st.sidebar.subheader("📊 Analysis Settings")

# Determine default value - only set on initial load
if 'analysis_periods' in st.session_state and st.session_state.analysis_periods is not None:
    # Widget already has a value, use it (prevents reset on rerun)
    default_periods = st.session_state.analysis_periods
else:
    # Initial default
    default_periods = ['1Y', '3Y']

# Callback function
def on_analysis_periods_change():
    """Update session state when periods are selected"""
    pass

analysis_periods = st.sidebar.multiselect(
    "Performance Periods:",
    options=['1Y', '3Y', '5Y', 'Inception'],
    default=default_periods,
    help="Select time periods for performance comparison",
    key="analysis_periods",
    on_change=on_analysis_periods_change
)

rolling_period = st.sidebar.selectbox(
    "Rolling CAGR Period:",
    options=[1, 3, 4, 5],
    index=2,
    help="Select rolling period for CAGR calculation"
)

# Load filtered data
@st.cache_data
def load_filtered_data(asset_codes, start_dt, end_dt):
    try:
        price_data, returns_data = calc.get_returns_data(asset_codes)

        # Filter by date range
        if start_dt:
            price_data = price_data[price_data.index >= pd.to_datetime(start_dt)]
            returns_data = returns_data[returns_data.index >= pd.to_datetime(start_dt)]
        if end_dt:
            price_data = price_data[price_data.index <= pd.to_datetime(end_dt)]
            returns_data = returns_data[returns_data.index <= pd.to_datetime(end_dt)]

        return price_data, returns_data
    except Exception as e:
        st.error(f"Error loading filtered data: {e}")
        return None, None

price_data, returns_data = load_filtered_data(selected_assets, start_date, end_date)

if price_data is None:
    st.stop()

# Filter summary
st.subheader("🎯 Filter Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Selected Assets", len(selected_assets))
with col2:
    st.metric("Date Range", f"{(end_date - start_date).days} days")
with col3:
    st.metric("From Date", start_date.strftime('%d/%m/%Y'))
with col4:
    st.metric("To Date", end_date.strftime('%d/%m/%Y'))

# Asset breakdown
fund_count = len([asset for asset in selected_assets if asset in fund_assets])
etf_count = len([asset for asset in selected_assets if asset in etf_assets])
benchmark_count = len([asset for asset in selected_assets if asset in benchmark_assets])

st.markdown(f"**Assets:** {fund_count} Funds, {etf_count} ETFs, {benchmark_count} Benchmarks")
st.markdown(f"**Selected Assets:** {', '.join(selected_assets)}")

# Matched Data Rows
st.subheader("📊 Filtered Data Rows")

try:
    # Get raw price data for the filtered period
    raw_data = processor.get_price_data(selected_assets, start_date, end_date)

    if not raw_data.empty:
        # Format the data for display
        display_data = raw_data.copy()
        display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
        display_data['price'] = display_data['price'].round(4)

        # Add asset type column
        display_data['asset_type'] = display_data['asset_code'].apply(
            lambda x: 'Fund' if x in fund_assets else ('ETF' if x in etf_assets else 'Benchmark')
        )

        # Reorder columns for better display
        display_data = display_data[['asset_code', 'asset_type', 'date', 'price']]
        display_data.columns = ['Asset Code', 'Type', 'Date', 'Price']

        # Sort by asset and date
        display_data = display_data.sort_values(['Asset Code', 'Date'])

        # Limit to top 2000 rows for display
        total_rows = len(display_data)
        display_limited = display_data.head(2000)

        # Display metrics about the filtered data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{total_rows:,}")
        with col2:
            st.metric("Displayed Rows", f"{len(display_limited):,}")
        with col3:
            st.metric("Unique Assets", display_data['Asset Code'].nunique())
        with col4:
            st.metric("Date Range", f"{(pd.to_datetime(end_date) - pd.to_datetime(start_date)).days} days")

        # Show warning if data is truncated
        if total_rows > 2000:
            st.warning(f"⚠️ Displaying top 2000 rows out of {total_rows:,} total rows. Download the full dataset using the button below.")

        # Display the data table (limited to 2000 rows)
        st.dataframe(
            display_limited,
            use_container_width=True,
            hide_index=True,
            height=500
        )

        # Download option
        st.markdown("#### Export Filtered Data")
        csv_data = display_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name=f"filtered_data_{start_date}_{end_date}.csv",
            mime="text/csv"
        )

    else:
        st.warning("No data found for the selected filters.")

except Exception as e:
    st.error(f"Error loading filtered data: {e}")

# Main analysis tabs
st.subheader("📈 Filtered Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance Comparison",
    "🔄 Rolling Analysis",
    "📊 Risk Analysis",
    "🔗 Correlation",
    "📋 Detailed Stats"
])

with tab1:
    # Cumulative returns comparison
    st.markdown("### Cumulative Performance Comparison")

    try:
        if not price_data.empty:
            # Normalize to 100% start for fair comparison
            cumulative_returns = (price_data / price_data.iloc[0] * 100).dropna()

            fig_cum = go.Figure()

            for i, asset in enumerate(cumulative_returns.columns):
                data = cumulative_returns[asset].dropna()
                if len(data) > 0:
                    fig_cum.add_trace(go.Scatter(
                        x=data.index,
                        y=data.values,
                        mode='lines',
                        name=asset,
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=3),
                        hovertemplate=f'<b>{asset}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Value: %{y:.1f}%<br>' +
                                    '<extra></extra>'
                    ))

            fig_cum.update_layout(
                title="Cumulative Returns Comparison (Normalized to 100%)",
                xaxis_title="Date",
                yaxis_title="Cumulative Value (%)",
                hovermode="x unified",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            fig_cum.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_cum, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating performance comparison: {e}")

with tab2:
    # Rolling CAGR analysis
    st.markdown(f"### Rolling {rolling_period}-Year CAGR Analysis")

    try:
        rolling_cagr = calc.calculate_rolling_cagr(price_data, years=rolling_period)

        if not rolling_cagr.empty:
            fig_rolling = go.Figure()

            for i, asset in enumerate(rolling_cagr.columns):
                data = rolling_cagr[asset].dropna()
                if len(data) > 0:
                    # Create custom hover text
                    hover_text = []
                    for date, cagr_value in zip(data.index, data.values):
                        start_date = date - timedelta(days=rolling_period*365)
                        hover_text.append(
                            f"<b>{asset}</b><br>" +
                            f"From: {start_date.strftime('%d/%m/%Y')} To: {date.strftime('%d/%m/%Y')}<br>" +
                            f"CAGR: {cagr_value:.1f}%"
                        )

                    fig_rolling.add_trace(go.Scatter(
                        x=data.index,
                        y=data.values,
                        mode='lines',
                        name=asset,
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=3),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_text
                    ))

            fig_rolling.update_layout(
                title=f"Rolling {rolling_period}-Year CAGR Comparison",
                xaxis_title="Date",
                yaxis_title="CAGR (%)",
                hovermode="x unified",
                height=500
            )

            fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_rolling, use_container_width=True)

            # Rolling CAGR statistics
            st.markdown("#### Rolling CAGR Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Summary Statistics**")
                rolling_stats = []
                for asset in rolling_cagr.columns:
                    data = rolling_cagr[asset].dropna()
                    if len(data) > 0:
                        rolling_stats.append({
                            'Asset': asset,
                            'Mean CAGR': f"{data.mean():.1f}%",
                            'Median CAGR': f"{data.median():.1f}%",
                            'Std Dev': f"{data.std():.1f}%",
                            'Min CAGR': f"{data.min():.1f}%",
                            'Max CAGR': f"{data.max():.1f}%"
                        })

                if rolling_stats:
                    rolling_stats_df = pd.DataFrame(rolling_stats)
                    st.dataframe(rolling_stats_df, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("**Best Performance Frequency**")
                best_performance_stats = []

                rolling_cagr_clean = rolling_cagr.dropna(how='all')

                for asset in rolling_cagr.columns:
                    best_count = 0
                    total_periods = 0

                    for date in rolling_cagr_clean.index:
                        period_data = rolling_cagr_clean.loc[date].dropna()

                        if asset in period_data.index and len(period_data) >= 2:
                            total_periods += 1
                            asset_value = period_data[asset]

                            if asset_value == period_data.max():
                                max_count = (period_data == period_data.max()).sum()
                                best_count += 1 / max_count

                    if total_periods > 0:
                        best_rate = (best_count / total_periods) * 100
                        best_performance_stats.append({
                            'Asset': asset,
                            'Best Rate': f"{best_rate:.1f}%",
                            'Total Periods': total_periods
                        })

                if best_performance_stats:
                    best_perf_df = pd.DataFrame(best_performance_stats)
                    st.dataframe(best_perf_df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"Not enough data for {rolling_period}-year rolling CAGR calculation.")

    except Exception as e:
        st.error(f"Error creating rolling analysis: {e}")

with tab3:
    # Risk analysis
    st.markdown("### Risk Analysis")

    try:
        if analysis_periods:
            period = analysis_periods[0]

            if period in summary:
                risk_return_data = []

                for asset in selected_assets:
                    ret = summary[period]['annualized_return'].get(asset)
                    vol = summary[period]['volatility'].get(asset)

                    if isinstance(ret, (int, float)) and isinstance(vol, (int, float)):
                        asset_type = 'Fund' if asset in fund_assets else ('ETF' if asset in etf_assets else 'Benchmark')
                        risk_return_data.append({
                            'Asset': asset,
                            'Return': ret,
                            'Risk': vol,
                            'Type': asset_type
                        })

                if risk_return_data:
                    rr_df = pd.DataFrame(risk_return_data)

                    # Risk-return scatter plot with asset type color coding
                    fig_rr = px.scatter(
                        rr_df,
                        x='Risk',
                        y='Return',
                        text='Asset',
                        color='Type',
                        title=f"Risk vs Return Analysis ({period})",
                        labels={'Risk': 'Volatility (%)', 'Return': 'Annualized Return (%)'},
                        color_discrete_map={'Fund': CHART_COLORS[0], 'Benchmark': CHART_COLORS[1]}
                    )

                    fig_rr.update_traces(textposition='top center', marker_size=12)
                    fig_rr.update_layout(height=500)

                    st.plotly_chart(fig_rr, use_container_width=True)

                    # Risk metrics table
                    st.markdown("#### Risk Metrics Summary")
                    risk_metrics = []

                    for asset in selected_assets:
                        max_dd = summary['max_drawdown'].get(asset, 'N/A')
                        sharpe = summary[period]['sharpe_ratio'].get(asset, 'N/A') if period in summary else 'N/A'
                        vol = summary[period]['volatility'].get(asset, 'N/A') if period in summary else 'N/A'

                        risk_metrics.append({
                            'Asset': asset,
                            'Type': 'Fund' if asset in fund_assets else ('ETF' if asset in etf_assets else 'Benchmark'),
                            'Volatility': f"{vol:.1f}%" if isinstance(vol, (int, float)) else 'N/A',
                            'Max Drawdown': f"{max_dd:.1f}%" if isinstance(max_dd, (int, float)) else 'N/A',
                            'Sharpe Ratio': f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else 'N/A'
                        })

                    risk_df = pd.DataFrame(risk_metrics)
                    st.dataframe(risk_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating risk analysis: {e}")

with tab4:
    # Correlation analysis
    st.markdown("### Correlation Analysis")

    try:
        if 'correlation' in summary and not summary['correlation'].empty:
            corr_matrix = summary['correlation']

            # Filter correlation matrix to selected assets
            available_assets = [asset for asset in selected_assets if asset in corr_matrix.index]
            if available_assets:
                filtered_corr = corr_matrix.loc[available_assets, available_assets]

                fig_corr = go.Figure(data=go.Heatmap(
                    z=filtered_corr.values,
                    x=filtered_corr.columns,
                    y=filtered_corr.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=filtered_corr.round(3).values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
                ))

                fig_corr.update_layout(
                    title="Asset Correlation Matrix",
                    height=max(400, len(available_assets) * 40)
                )

                st.plotly_chart(fig_corr, use_container_width=True)

                # Correlation insights
                st.markdown("#### Correlation Insights")

                # Find highest and lowest correlations
                corr_pairs = []
                for i in range(len(available_assets)):
                    for j in range(i+1, len(available_assets)):
                        asset1 = available_assets[i]
                        asset2 = available_assets[j]
                        corr_val = filtered_corr.loc[asset1, asset2]
                        corr_pairs.append({
                            'Asset 1': asset1,
                            'Asset 2': asset2,
                            'Correlation': corr_val
                        })

                if corr_pairs:
                    corr_df = pd.DataFrame(corr_pairs)
                    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Highest Correlations**")
                        high_corr = corr_df.head(5).copy()
                        high_corr['Correlation'] = high_corr['Correlation'].apply(lambda x: f"{x:.3f}")
                        st.dataframe(high_corr, use_container_width=True, hide_index=True)

                    with col2:
                        st.markdown("**Lowest Correlations**")
                        low_corr = corr_df.tail(5).copy()
                        low_corr['Correlation'] = low_corr['Correlation'].apply(lambda x: f"{x:.3f}")
                        st.dataframe(low_corr, use_container_width=True, hide_index=True)
        else:
            st.warning("Correlation data not available for selected assets.")

    except Exception as e:
        st.error(f"Error creating correlation analysis: {e}")

with tab5:
    # Detailed statistics
    st.markdown("### Detailed Statistics")

    try:
        # Calculate detailed stats for each asset
        detailed_stats = []

        for asset in selected_assets:
            if asset in price_data.columns:
                asset_prices = price_data[asset].dropna()

                if len(asset_prices) > 1:
                    # Basic price stats
                    start_price = asset_prices.iloc[0]
                    end_price = asset_prices.iloc[-1]
                    total_return = ((end_price - start_price) / start_price) * 100

                    # Calculate annualized return
                    days_invested = (asset_prices.index[-1] - asset_prices.index[0]).days
                    years_invested = days_invested / 365.25
                    annualized_return = ((end_price / start_price) ** (1/years_invested) - 1) * 100 if years_invested > 0 else 0

                    # Returns analysis
                    returns = asset_prices.pct_change().dropna()
                    daily_vol = returns.std() * 100
                    annual_vol = daily_vol * np.sqrt(252)

                    # Drawdown
                    peak = asset_prices.expanding(min_periods=1).max()
                    drawdown = ((asset_prices - peak) / peak * 100)
                    max_drawdown = drawdown.min()

                    detailed_stats.append({
                        'Asset': asset,
                        'Type': 'Fund' if asset in fund_assets else ('ETF' if asset in etf_assets else 'Benchmark'),
                        'Start Price': f"{start_price:.2f}",
                        'End Price': f"{end_price:.2f}",
                        'Total Return': f"{total_return:.1f}%",
                        'Annual Return': f"{annualized_return:.1f}%",
                        'Annual Volatility': f"{annual_vol:.1f}%",
                        'Max Drawdown': f"{max_drawdown:.1f}%",
                        'Data Points': len(asset_prices),
                        'Date Range': f"{days_invested} days"
                    })

        if detailed_stats:
            detailed_df = pd.DataFrame(detailed_stats)
            st.dataframe(detailed_df, use_container_width=True)

            # Export option
            st.markdown("#### Export Data")
            col1, col2 = st.columns(2)

            with col1:
                csv_data = detailed_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Detailed Stats as CSV",
                    data=csv_data,
                    file_name=f"filtered_analysis_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )

            with col2:
                summary_csv = summary_df.to_csv(index=False) if 'summary_df' in locals() else ""
                if summary_csv:
                    st.download_button(
                        label="📥 Download Summary as CSV",
                        data=summary_csv,
                        file_name=f"performance_summary_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Error generating detailed statistics: {e}")

# Footer
st.markdown("---")
st.markdown("**📊 Price Data Listing** | [Dashboard](/) | [Metrics Guide](/1_📚_Metrics_Guide) | [Asset Detail](/2_📊_Asset_Detail)")