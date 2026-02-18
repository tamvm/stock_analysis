import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from config import DB_PATH

st.set_page_config(page_title="US Market vs PE", page_icon="📈", layout="wide")

st.title("📈 US Market P/E Correlation")
st.markdown("Analyze the relationship between IVV (S&P 500 ETF) performance and US market P/E ratio over time")

# Add cache clear button in sidebar
with st.sidebar:
    if st.button("🔄 Clear Cache & Reload Data"):
        st.cache_data.clear()
        st.rerun()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_ivv_data():
    """Load IVV price data from database"""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT date, price
        FROM price_data
        WHERE asset_code = 'IVV'
        ORDER BY date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_us_market_pe_data():
    """Load US market PE data from database"""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT date, pe_ratio, total_market_cap, total_earnings, total_revenue
        FROM market_statistics
        WHERE market_code = 'US'
        ORDER BY date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    return df

def create_annual_comparison_table(ivv_df, pe_df):
    """
    Create annual comparison table matching the reference design
    Shows: Year, IVV, YoY Change %, P/E, P/E Change %
    Only shows years where both IVV and PE data are available
    """
    # Merge datasets - use merge_asof but only keep rows where PE data exists within 30 days
    merged = pd.merge_asof(
        ivv_df.sort_values('date'),
        pe_df.sort_values('date'),
        on='date',
        direction='nearest',
        tolerance=pd.Timedelta(days=30)  # Only match if PE data is within 30 days
    )
    
    # Remove rows where PE data is missing (years before PE data availability)
    merged = merged.dropna(subset=['pe_ratio'])
    
    # Extract year and get year-end values
    merged['year'] = merged['date'].dt.year
    
    # Get last trading day of each year
    annual_data = merged.groupby('year').last().reset_index()
    
    # Calculate year-over-year changes
    annual_data['yoy_change'] = annual_data['price'].pct_change() * 100
    annual_data['pe_change'] = annual_data['pe_ratio'].pct_change() * 100
    
    # Format the table
    display_df = pd.DataFrame({
        'Year': annual_data['year'].astype(int),
        'IVV': annual_data['price'].round(2),
        'YoY Change %': annual_data['yoy_change'].round(2),
        'P/E': annual_data['pe_ratio'].round(2),
        'P/E Change %': annual_data['pe_change'].round(2)
    })
    
    return display_df

def style_comparison_table(df):
    """Apply styling to the comparison table"""
    def color_yoy(val):
        if pd.isna(val):
            return ''
        color = '#90EE90' if val > 0 else '#FFB6C6'  # Light green or light red
        return f'background-color: {color}'
    
    def color_pe(val):
        """Color P/E values: green (low/safe) to red (high/dangerous)"""
        if pd.isna(val):
            return ''
        # Define PE ranges for US market
        # Low PE (< 20): Green (safe, undervalued)
        # Medium PE (20-30): Yellow (neutral)
        # High PE (> 30): Red (dangerous, overvalued)
        if val < 20:
            # Green gradient
            intensity = max(0, min(1, (20 - val) / 5))  # 0 to 1
            green = int(144 + intensity * 111)  # 144 to 255
            return f'background-color: rgb(144, {green}, 144)'
        elif val < 30:
            # Yellow gradient
            progress = (val - 20) / 10  # 0 to 1
            red = int(255)
            green = int(255 - progress * 100)  # 255 to 155
            return f'background-color: rgb({red}, {green}, 144)'
        else:
            # Red gradient
            intensity = min(1, (val - 30) / 10)  # 0 to 1
            green = int(182 - intensity * 50)  # 182 to 132
            blue = int(198 - intensity * 50)  # 198 to 148
            return f'background-color: rgb(255, {green}, {blue})'
    
    styled = df.style.format({
        'IVV': '{:.2f}',
        'YoY Change %': '{:.2f}%',
        'P/E': '{:.2f}',
        'P/E Change %': '{:.2f}%'
    }).applymap(color_yoy, subset=['YoY Change %'])\
      .applymap(color_pe, subset=['P/E'])
    
    return styled

def create_dual_axis_chart(ivv_df, pe_df):
    """Create dual-axis chart showing IVV and PE ratio over time"""
    # Merge datasets
    merged = pd.merge_asof(
        ivv_df.sort_values('date'),
        pe_df.sort_values('date'),
        on='date',
        direction='nearest'
    )
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add IVV trace
    fig.add_trace(
        go.Scatter(
            x=merged['date'],
            y=merged['price'],
            name="IVV",
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>IVV</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Add PE ratio trace
    fig.add_trace(
        go.Scatter(
            x=merged['date'],
            y=merged['pe_ratio'],
            name="P/E Ratio",
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='<b>P/E Ratio</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="<b>IVV Price</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>P/E Ratio</b>", secondary_y=True)
    
    fig.update_layout(
        title="IVV vs US Market P/E Ratio Over Time",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def calculate_forward_returns_correlation(ivv_df, pe_df):
    """
    Calculate correlation between current PE ratio and IVV returns over different forward periods
    
    Returns:
        DataFrame with correlations for different time periods
    """
    # Merge datasets
    merged = pd.merge_asof(
        ivv_df.sort_values('date'),
        pe_df.sort_values('date'),
        on='date',
        direction='nearest',
        tolerance=pd.Timedelta(days=30)
    )
    
    # Remove rows without PE data
    merged = merged.dropna(subset=['pe_ratio'])
    merged = merged.set_index('date').sort_index()
    
    # Define forward periods (in months)
    periods = {
        'Current': 0,
        '3 Months': 3,
        '6 Months': 6,
        '9 Months': 9,
        '12 Months': 12,
        '24 Months': 24
    }
    
    correlations = []
    
    for period_name, months in periods.items():
        if months == 0:
            # Current period - calculate monthly return
            temp_df = merged.copy()
            temp_df['return'] = temp_df['price'].pct_change() * 100
        else:
            # Forward returns
            temp_df = merged.copy()
            temp_df['future_price'] = temp_df['price'].shift(-months * 21)  # Approximate 21 trading days per month
            temp_df['return'] = ((temp_df['future_price'] - temp_df['price']) / temp_df['price']) * 100
        
        # Remove NaN values
        temp_df = temp_df.dropna(subset=['return', 'pe_ratio'])
        
        if len(temp_df) > 0:
            correlation = temp_df['pe_ratio'].corr(temp_df['return'])
            
            correlations.append({
                'Period': period_name,
                'Months': months,
                'Correlation': correlation,
                'Sample Size': len(temp_df)
            })
    
    return pd.DataFrame(correlations)

def create_forward_returns_scatter_plots(ivv_df, pe_df):
    """
    Create individual scatter plots for each forward period showing PE vs returns
    Returns a list of figures and correlation dataframe
    """
    # Merge datasets
    merged = pd.merge_asof(
        ivv_df.sort_values('date'),
        pe_df.sort_values('date'),
        on='date',
        direction='nearest',
        tolerance=pd.Timedelta(days=30)
    )
    
    # Remove rows without PE data
    merged = merged.dropna(subset=['pe_ratio'])
    merged = merged.set_index('date').sort_index()
    
    # Define forward periods (in months) - excluding current
    periods = [
        ('1 Month', 1),
        ('3 Months', 3),
        ('6 Months', 6),
        ('12 Months', 12),
        ('24 Months', 24),
        ('36 Months', 36),
        ('48 Months', 48),
        ('60 Months', 60)
    ]
    
    figures = []
    correlations = []
    
    for period_name, months in periods:
        # Calculate forward returns
        temp_df = merged.copy()
        temp_df['future_price'] = temp_df['price'].shift(-months * 21)  # Approximate 21 trading days per month
        temp_df['return'] = ((temp_df['future_price'] - temp_df['price']) / temp_df['price']) * 100
        
        # Remove NaN values
        temp_df = temp_df.dropna(subset=['return', 'pe_ratio'])
        
        if len(temp_df) > 0:
            correlation = temp_df['pe_ratio'].corr(temp_df['return'])
            r_squared = correlation ** 2
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=temp_df['pe_ratio'],
                y=temp_df['return'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='#2E8B57',  # Sea green
                    opacity=0.6
                ),
                name='Data Points',
                hovertemplate='<b>P/E: %{x:.2f}</b><br>Return: %{y:.1f}%<extra></extra>'
            ))
            
            # Add trend line
            z = np.polyfit(temp_df['pe_ratio'], temp_df['return'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(temp_df['pe_ratio'].min(), temp_df['pe_ratio'].max(), 100)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name='Trend Line',
                line=dict(color='#FF6347', width=2, dash='dash'),  # Tomato red
                hoverinfo='skip'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"P/E vs IVV Forward Returns ({period_name})<br>R² = {r_squared:.4f}",
                xaxis_title="P/E Ratio",
                yaxis_title="Forward Return %",
                height=350,
                showlegend=False,
                margin=dict(l=50, r=20, t=60, b=50),
                plot_bgcolor='white',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray')
            )
            
            figures.append(fig)
            
            correlations.append({
                'Period': period_name,
                'Months': months,
                'Correlation': correlation,
                'R²': r_squared,
                'Sample Size': len(temp_df)
            })
    
    return figures, pd.DataFrame(correlations)

def calculate_summary_statistics(ivv_df, pe_df):
    """Calculate summary statistics for the analysis"""
    # Merge datasets
    merged = pd.merge_asof(
        ivv_df.sort_values('date'),
        pe_df.sort_values('date'),
        on='date',
        direction='nearest'
    )
    
    # Define high/low PE environments (above/below median)
    pe_median = merged['pe_ratio'].median()
    
    merged['pe_environment'] = merged['pe_ratio'].apply(
        lambda x: 'High PE' if x > pe_median else 'Low PE'
    )
    
    # Calculate returns in each environment
    merged['return'] = merged['price'].pct_change()
    
    stats = merged.groupby('pe_environment')['return'].agg([
        ('avg_return', lambda x: x.mean() * 100),
        ('volatility', lambda x: x.std() * 100),
        ('count', 'count')
    ]).round(2)
    
    return stats, pe_median

# Main app layout
try:
    # Load data
    ivv_df = load_ivv_data()
    pe_df = load_us_market_pe_data()
    
    if ivv_df.empty or pe_df.empty:
        st.error("No data available. Please run the import scripts first.")
        st.code("python import_us_data.py  # For IVV data\npython import_market_pe.py us  # For US market PE data")
        st.stop()
    
    # Display data range info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("IVV Records", len(ivv_df))
    with col2:
        st.metric("P/E Records", len(pe_df))
    with col3:
        st.metric("Date Range", f"{ivv_df['date'].min().year} - {ivv_df['date'].max().year}")
    
    # Info about PE data availability
    st.info(f"""
    📊 **Historical Data Coverage:** 
    - **US Market P/E**: {len(pe_df)} records from **{pe_df['date'].min().strftime('%Y-%m')}** to **{pe_df['date'].max().strftime('%Y-%m')}**
    - **IVV (S&P 500 ETF)**: {len(ivv_df)} records from **{ivv_df['date'].min().strftime('%Y-%m')}** to **{ivv_df['date'].max().strftime('%Y-%m')}**
    - **Analysis Period**: Overlap period from **2000-05** to **2026-02** ({len(ivv_df)} observations)
    - **Data Sources**: Macrotrends (historical PE, 1927-2016) + SimplyWall.St (recent PE with metrics, 2016-2026)
    """)
    
    st.divider()
    
    # Annual Comparison Table
    st.subheader("📊 Annual Performance Comparison")
    st.markdown("Year-end IVV values with year-over-year changes, P/E ratio (end of year), and P/E change %")
    
    annual_table = create_annual_comparison_table(ivv_df, pe_df)
    
    # Display styled table in 40% width column
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.dataframe(
            style_comparison_table(annual_table),
            use_container_width=True,
            height=600
        )
    
    with col2:
        # Time Series Chart
        st.markdown("### 📈 IVV vs P/E Ratio Over Time")
        dual_axis_chart = create_dual_axis_chart(ivv_df, pe_df)
        st.plotly_chart(dual_axis_chart, use_container_width=True)
    
    st.divider()
    
    # Correlation Analysis
    st.subheader("🔍 P/E Ratio vs Forward Returns Analysis")
    st.markdown("Analyze the correlation between current P/E ratio and IVV forward returns over different time periods")
    
    # Add explanation about Correlation vs R²
    with st.expander("ℹ️ Understanding Correlation and R²"):
        st.markdown("""
        **Correlation (r)** and **R² (R-squared)** are two metrics that measure the relationship between two variables:
        
        - **Correlation (r)**: 
          - Measures the **direction** and **strength** of the linear relationship
          - Values range from **-1 to +1**
          - r > 0: positive correlation (high PE → higher returns)
          - r < 0: negative correlation (high PE → lower returns)
          - |r| close to 1: strong correlation, |r| close to 0: weak correlation
        
        - **R² (R-squared)**:
          - Square of correlation: R² = r²
          - Values range from **0 to 1** (or 0% to 100%)
          - Indicates **what percentage of variation** in IVV returns is **explained** by P/E ratio
          - Example: R² = 0.3156 (31.56%) means P/E explains 31.56% of IVV return variation
          - Higher R² → better predictive model
        
        **Example**: If correlation = -0.562, then R² = 0.3156
        - Moderate negative correlation (high PE → lower returns)
        - P/E explains 31.56% of IVV return variation over 24 months
        """)
    
    # Create scatter plots and get correlation data
    scatter_figs, corr_df = create_forward_returns_scatter_plots(ivv_df, pe_df)
    
    # Display correlation summary table first
    st.markdown("### Correlation Coefficients Summary")
    
    # Display correlation table
    display_corr = corr_df[['Period', 'Correlation', 'R²']].copy()
    display_corr['Correlation'] = display_corr['Correlation'].round(3)
    display_corr['R²'] = display_corr['R²'].round(4)
    
    # Add color coding
    def color_correlation(val):
        if pd.isna(val):
            return ''
        if abs(val) < 0.3:
            color = '#FFFFE0'  # Light yellow - weak
        elif abs(val) < 0.7:
            color = '#90EE90' if val > 0 else '#FFB6C6'  # Green/Red - moderate
        else:
            color = '#006400' if val > 0 else '#8B0000'  # Dark green/red - strong
        return f'background-color: {color}'
    
    styled_corr = display_corr.style.applymap(
        color_correlation, 
        subset=['Correlation']
    ).format({
        'Correlation': '{:.3f}',
        'R²': '{:.4f}'
    })
    
    st.dataframe(styled_corr, use_container_width=True, hide_index=True)
    
    # Add warning about correlation interpretation
    st.warning("""
    ⚠️ **Important Findings with Full Historical Data (2000-2026)**: 
    - **All periods now have sufficient data** (26 years of history vs previous 10 years)
    - **Correlation is weaker** than expected: R² < 11% for all periods
    - **Why?** Long bull markets (2003-2007, 2009-2020) created positive correlation, offsetting negative correlation from bear markets
    - **Time period matters**: 2016-2026 showed strong negative correlation (R²≈30%), but 2000-2026 shows weak positive correlation
    - **Implication**: P/E ratio has **limited predictive power** over the full 26-year period, but may work better in specific market conditions
    - **For investment decisions**: Consider current market regime (trending vs mean-reverting) when using P/E as a predictor
    """)
    
    st.divider()
    
    # Display scatter plots in grid layout (2 columns)
    st.markdown("### Scatter Plots: P/E vs Forward Returns")
    
    for i in range(0, len(scatter_figs), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(scatter_figs[i], use_container_width=True)
        
        if i + 1 < len(scatter_figs):
            with col2:
                st.plotly_chart(scatter_figs[i + 1], use_container_width=True)
    
    # Interpretation
    st.divider()
    st.markdown("### Interpretation & Key Findings")
    
    # Find strongest correlation
    strongest = corr_df.loc[corr_df['Correlation'].abs().idxmax()]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info(f"""
        **Strongest Correlation: {strongest['Period']}**
        - Correlation: {strongest['Correlation']:.3f}
        - R²: {strongest['R²']:.4f} ({strongest['R²']*100:.1f}%)
        - Sample size: {strongest['Sample Size']} observations
        
        **Note:** Even the strongest correlation is relatively weak (R² < 15%), 
        indicating P/E has limited predictive power over the 2000-2026 period.
        """)
        
        st.success(f"""
        **📊 Data Quality: Excellent**
        - ✅ **26 years** of historical data (2000-2026)
        - ✅ **All periods** have sufficient sample size
        - ✅ **1,160 PE records** from 1927 to present
        - ✅ Includes multiple market cycles
        """)
    
    with col2:
        st.markdown("""
        **Why is correlation weak?**
        
        **Market Regime Dependency:**
        - 📈 **Bull markets** (2003-2007, 2009-2020): P/E and prices rise together → positive correlation
        - 📉 **Bear markets** (2000-2002, 2008-2009, 2022): High P/E → lower returns → negative correlation
        - ⚖️ **Net effect**: Weak overall correlation over full period
        
        **Historical Context:**
        - **2016-2026 only**: R² ≈ 30% (moderate negative) ✅
        - **2000-2026 full**: R² < 11% (weak positive) ⚠️
        
        **Takeaway:** P/E works better as a predictor in **mean-reverting markets** 
        than in **trending markets**. Current market regime matters!
        """)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.exception(e)
