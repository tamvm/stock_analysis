import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from config import DB_PATH

st.set_page_config(page_title="VNIndex vs Market PE", page_icon="📈", layout="wide")

st.title("📈 VNIndex vs Market P/E Correlation")
st.markdown("Analyze the relationship between VNIndex performance and Vietnam market P/E ratio over time")

@st.cache_data
def load_vnindex_data():
    """Load VNIndex price data from database"""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT date, price
        FROM price_data
        WHERE asset_code = 'VNINDEX'
        ORDER BY date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_market_pe_data():
    """Load Vietnam market PE data from database"""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT date, pe_ratio, total_market_cap, total_earnings, total_revenue
        FROM market_statistics
        WHERE market_code = 'VN'
        ORDER BY date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    return df

def create_annual_comparison_table(vnindex_df, pe_df):
    """
    Create annual comparison table matching the reference design
    Shows: Year, VNIndex, YoY Change %, P/E, P/E Change %
    Only shows years where both VNIndex and PE data are available
    """
    # Merge datasets - use merge_asof but only keep rows where PE data exists within 30 days
    merged = pd.merge_asof(
        vnindex_df.sort_values('date'),
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
        'VNIndex': annual_data['price'].round(2),
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
        # Define PE ranges for Vietnam market
        # Low PE (< 12): Green (safe, undervalued)
        # Medium PE (12-16): Yellow (neutral)
        # High PE (> 16): Red (dangerous, overvalued)
        if val < 12:
            # Green gradient
            intensity = max(0, min(1, (12 - val) / 4))  # 0 to 1
            green = int(144 + intensity * 111)  # 144 to 255
            return f'background-color: rgb(144, {green}, 144)'
        elif val < 16:
            # Yellow gradient
            progress = (val - 12) / 4  # 0 to 1
            red = int(255)
            green = int(255 - progress * 100)  # 255 to 155
            return f'background-color: rgb({red}, {green}, 144)'
        else:
            # Red gradient
            intensity = min(1, (val - 16) / 4)  # 0 to 1
            green = int(182 - intensity * 50)  # 182 to 132
            blue = int(198 - intensity * 50)  # 198 to 148
            return f'background-color: rgb(255, {green}, {blue})'
    
    styled = df.style.format({
        'VNIndex': '{:.2f}',
        'YoY Change %': '{:.2f}%',
        'P/E': '{:.2f}',
        'P/E Change %': '{:.2f}%'
    }).applymap(color_yoy, subset=['YoY Change %'])\
      .applymap(color_pe, subset=['P/E'])
    
    return styled

def create_dual_axis_chart(vnindex_df, pe_df):
    """Create dual-axis chart showing VNIndex and PE ratio over time"""
    # Merge datasets
    merged = pd.merge_asof(
        vnindex_df.sort_values('date'),
        pe_df.sort_values('date'),
        on='date',
        direction='nearest'
    )
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add VNIndex trace
    fig.add_trace(
        go.Scatter(
            x=merged['date'],
            y=merged['price'],
            name="VNIndex",
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>VNIndex</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
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
    fig.update_yaxes(title_text="<b>VNIndex</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>P/E Ratio</b>", secondary_y=True)
    
    fig.update_layout(
        title="VNIndex vs Market P/E Ratio Over Time",
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

def calculate_forward_returns_correlation(vnindex_df, pe_df):
    """
    Calculate correlation between current PE ratio and VNIndex returns over different forward periods
    
    Returns:
        DataFrame with correlations for different time periods
    """
    # Merge datasets
    merged = pd.merge_asof(
        vnindex_df.sort_values('date'),
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

def create_forward_returns_scatter_plots(vnindex_df, pe_df):
    """
    Create individual scatter plots for each forward period showing PE vs returns
    Returns a list of figures and correlation dataframe
    """
    # Merge datasets
    merged = pd.merge_asof(
        vnindex_df.sort_values('date'),
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
        ('9 Months', 9),
        ('12 Months', 12),
        ('24 Months', 24),
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
                title=f"P/E và tăng trưởng VNIndex {period_name.lower()} tiếp theo<br>R² = {r_squared:.4f}",
                xaxis_title="P/E",
                yaxis_title="Tăng trưởng",
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

def calculate_summary_statistics(vnindex_df, pe_df):
    """Calculate summary statistics for the analysis"""
    # Merge datasets
    merged = pd.merge_asof(
        vnindex_df.sort_values('date'),
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
    vnindex_df = load_vnindex_data()
    pe_df = load_market_pe_data()
    
    if vnindex_df.empty or pe_df.empty:
        st.error("No data available. Please run the import scripts first.")
        st.code("python import_vn_funds.py vnindex\npython import_market_pe.py vn")
        st.stop()
    
    # Display data range info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("VNIndex Records", len(vnindex_df))
    with col2:
        st.metric("P/E Records", len(pe_df))
    with col3:
        st.metric("Date Range", f"{vnindex_df['date'].min().year} - {vnindex_df['date'].max().year}")
    
    # Info about PE data availability
    st.info(f"📊 **Lưu ý:** Dữ liệu P/E thị trường VN chỉ có từ **{pe_df['date'].min().strftime('%Y-%m')}** đến **{pe_df['date'].max().strftime('%Y-%m')}** (nguồn: Simplize API). Các năm trước 2016 không có dữ liệu P/E.")
    
    st.divider()
    
    # Annual Comparison Table
    st.subheader("📊 Annual Performance Comparison")
    st.markdown("Year-end VNIndex values with year-over-year changes, P/E ratio (end of year), and P/E change %")
    
    annual_table = create_annual_comparison_table(vnindex_df, pe_df)
    
    # Display styled table in 40% width column
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.dataframe(
            style_comparison_table(annual_table),
            use_container_width=True,
            height=600
        )
    
    st.divider()
    
    # Time Series Chart
    st.subheader("📈 VNIndex vs P/E Ratio Over Time")
    dual_axis_chart = create_dual_axis_chart(vnindex_df, pe_df)
    st.plotly_chart(dual_axis_chart, use_container_width=True)
    
    st.divider()
    
    # Correlation Analysis
    st.subheader("🔍 P/E Ratio vs Forward Returns Analysis")
    st.markdown("Phân tích mối tương quan giữa P/E ratio hiện tại và tăng trưởng VNIndex trong tương lai")
    
    # Add explanation about Correlation vs R²
    with st.expander("ℹ️ Giải thích về Correlation và R²"):
        st.markdown("""
        **Correlation (r)** và **R² (R-squared)** là hai chỉ số đo lường mối quan hệ giữa hai biến số:
        
        - **Correlation (r)**: 
          - Đo lường **hướng** và **độ mạnh** của mối quan hệ tuyến tính
          - Giá trị từ **-1 đến +1**
          - r > 0: tương quan dương (PE cao → VNIndex tăng)
          - r < 0: tương quan âm (PE cao → VNIndex giảm)
          - |r| gần 1: tương quan mạnh, |r| gần 0: tương quan yếu
        
        - **R² (R-squared)**:
          - Bình phương của correlation: R² = r²
          - Giá trị từ **0 đến 1** (hoặc 0% đến 100%)
          - Cho biết **bao nhiêu % biến động** của VNIndex được **giải thích** bởi P/E ratio
          - VD: R² = 0.3156 (31.56%) nghĩa là P/E giải thích được 31.56% biến động của VNIndex
          - R² càng cao → mô hình dự đoán càng tốt
        
        **Ví dụ**: Nếu correlation = -0.562, thì R² = 0.3156
        - Tương quan âm trung bình (PE cao → VNIndex giảm)
        - P/E giải thích được 31.56% biến động VNIndex sau 24 tháng
        """)
    
    # Create scatter plots and get correlation data
    scatter_figs, corr_df = create_forward_returns_scatter_plots(vnindex_df, pe_df)
    
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
    st.markdown("### Interpretation")
    
    # Find strongest correlation
    strongest = corr_df.loc[corr_df['Correlation'].abs().idxmax()]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info(f"""
        **Mối tương quan mạnh nhất:** {strongest['Period']}
        - Correlation: {strongest['Correlation']:.3f}
        - R²: {strongest['R²']:.4f}
        - Sample size: {strongest['Sample Size']} observations
        """)
    
    with col2:
        st.markdown("""
        **Ý nghĩa:**
        - **Correlation > 0**: PE cao → VNIndex tăng
        - **Correlation < 0**: PE cao → VNIndex giảm
        - **R² gần 1**: Mô hình giải thích tốt
        - **|Correlation| < 0.3**: Tương quan yếu
        - **0.3 ≤ |Correlation| < 0.7**: Tương quan trung bình
        - **|Correlation| ≥ 0.7**: Tương quan mạnh
        """)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.exception(e)
