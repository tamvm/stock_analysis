import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
import numpy as np

# Page config
st.set_page_config(
    page_title="Top Performers by Year",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Asset color mapping based on fund characteristics
# Colors are inspired by common fund categorization:
# - Bond funds (DCDE, DCBF, VCBFTBF, etc.): Teal/Green shades
# - Equity funds (VESAF, VEOF, etc.): Red/Pink shades
# - Balanced funds (SSISCA, SCA, MGF, VLGF): Blue shades
# - Special funds (BVFED, ENF, BVPF, MAGEF): Gold/Beige/Gray shades
# - Indices (VNINDEX, VN30): White with border

ASSET_COLORS = {
    # Bond Funds - Teal/Green family
    'DCDE': '#0D9488',      # Teal-600
    'DCBF': '#14B8A6',      # Teal-500
    'DCDS': '#2DD4BF',      # Teal-400
    'VCBFTBF': '#5EEAD4',   # Teal-300
    'TBF': '#0891B2',       # Cyan-600
    
    # Equity Funds - Red/Pink family
    'VESAF': '#DC2626',     # Red-600
    'VEOF': '#EF4444',      # Red-500
    'VEASAF': '#B91C1C',    # Red-700
    
    # Balanced Funds - Blue family
    'SSISCA': '#1E3A8A',    # Blue-900
    'SCA': '#1E40AF',       # Blue-800
    'MGF': '#2563EB',       # Blue-600
    'VLGF': '#3B82F6',      # Blue-500
    'VCBFBCF': '#60A5FA',   # Blue-400
    
    # Special Funds - Gold/Beige/Gray family
    'BVFED': '#D97706',     # Amber-600
    'ENF': '#F59E0B',       # Amber-500
    'BVPF': '#FBBF24',      # Amber-400
    'NTPPF': '#FCD34D',     # Amber-300
    'MAGEF': '#6B7280',     # Gray-500
    'MAFEQI': '#9CA3AF',    # Gray-400
    'KDEF': '#A78BFA',      # Purple-400
    'UVEEF': '#C084FC',     # Purple-400
    'VCAMDF': '#E879F9',    # Fuchsia-400
    'VEMEEF': '#F472B6',    # Pink-400
    
    # Indices - White with border (will be styled differently)
    'VNINDEX': '#FFFFFF',   # White
    'VN30': '#F3F4F6',      # Gray-100
    
    # US ETFs
    'VOO': '#1F77B4',       # Blue
    'QQQ': '#FF7F0E',       # Orange
    'VTI': '#2CA02C',       # Green
    
    # Crypto
    'BTC': '#F7931A',       # Bitcoin Orange
    'ETH': '#627EEA',       # Ethereum Blue
    
    # Commodities
    'GOLD': '#FFD700',      # Gold
    'SILVER': '#C0C0C0',    # Silver
}

def get_asset_color(asset_code, index=0):
    """
    Get color for an asset. If not in predefined mapping, generate from palette.
    
    Args:
        asset_code: Asset code
        index: Fallback index for color palette
    
    Returns:
        Hex color code
    """
    if asset_code in ASSET_COLORS:
        return ASSET_COLORS[asset_code]
    
    # Fallback color palette
    fallback_colors = [
        '#8B5CF6', '#EC4899', '#10B981', '#F59E0B', '#3B82F6',
        '#EF4444', '#14B8A6', '#F97316', '#6366F1', '#84CC16'
    ]
    return fallback_colors[index % len(fallback_colors)]

def calculate_yearly_returns(db_path="db/investment_data.db"):
    """
    Calculate yearly returns for all assets from start to end of each year.
    
    Returns:
        DataFrame with columns: year, asset_code, yearly_return, asset_type
    """
    conn = sqlite3.connect(db_path)
    
    # Get all price data
    query = """
        SELECT 
            p.date,
            p.asset_code,
            p.price,
            a.asset_type
        FROM price_data p
        JOIN assets a ON p.asset_code = a.asset_code
        ORDER BY p.asset_code, p.date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Calculate yearly returns
    yearly_returns = []
    
    for asset in df['asset_code'].unique():
        asset_data = df[df['asset_code'] == asset].copy()
        asset_type = asset_data['asset_type'].iloc[0]
        
        for year in asset_data['year'].unique():
            year_data = asset_data[asset_data['year'] == year].sort_values('date')
            
            if len(year_data) < 2:
                continue
            
            # Get first and last price of the year
            start_price = year_data.iloc[0]['price']
            end_price = year_data.iloc[-1]['price']
            
            # Calculate return
            yearly_return = ((end_price - start_price) / start_price) * 100
            
            yearly_returns.append({
                'year': year,
                'asset_code': asset,
                'asset_type': asset_type,
                'yearly_return': yearly_return,
                'start_date': year_data.iloc[0]['date'],
                'end_date': year_data.iloc[-1]['date'],
                'start_price': start_price,
                'end_price': end_price
            })
    
    return pd.DataFrame(yearly_returns)

def get_top_performers(yearly_returns_df, asset_types, top_n=3):
    """
    Get top N performers for each year, filtered by asset types.
    
    Args:
        yearly_returns_df: DataFrame with yearly returns
        asset_types: List of asset types to include
        top_n: Number of top performers to return per year
    
    Returns:
        DataFrame with top performers
    """
    # Filter by asset types
    filtered_df = yearly_returns_df[yearly_returns_df['asset_type'].isin(asset_types)].copy()
    
    # Get top N performers for each year
    top_performers = []
    
    for year in sorted(filtered_df['year'].unique()):
        year_data = filtered_df[filtered_df['year'] == year].copy()
        year_data = year_data.sort_values('yearly_return', ascending=False).head(top_n)
        
        # Add rank
        year_data['rank'] = range(1, len(year_data) + 1)
        
        top_performers.append(year_data)
    
    if top_performers:
        return pd.concat(top_performers, ignore_index=True)
    else:
        return pd.DataFrame()

# Title
st.title("🏆 Top Performers by Year")
st.markdown("**Analyze the top 3 assets with highest annual growth for each year**")

# Sidebar - Asset Type Selection
st.sidebar.header("Filter Settings")

# Get available asset types
conn = sqlite3.connect("db/investment_data.db")
available_types_df = pd.read_sql_query(
    "SELECT DISTINCT asset_type FROM assets ORDER BY asset_type",
    conn
)
conn.close()

available_types = available_types_df['asset_type'].tolist()

# Asset type checkboxes
st.sidebar.subheader("📊 Asset Types")
st.sidebar.markdown("*Select asset types to include in analysis*")

selected_types = []

# Create a mapping of asset types to display names
asset_type_labels = {
    'vn_fund': '🇻🇳 VN Funds',
    'vn_index': '📈 VN Indices',
    'us_etf': '🇺🇸 US ETFs',
    'us_stock': '🇺🇸 US Stocks',
    'crypto': '₿ Crypto',
    'commodity': '🥇 Commodities',
    'benchmark': '📊 Benchmarks'
}

# Default to vn_fund
default_types = ['vn_fund']

for asset_type in available_types:
    label = asset_type_labels.get(asset_type, asset_type.replace('_', ' ').title())
    is_default = asset_type in default_types
    
    if st.sidebar.checkbox(label, value=is_default, key=f"type_{asset_type}"):
        selected_types.append(asset_type)

# Always include VNINDEX as reference if vn_index is available
include_vnindex = st.sidebar.checkbox(
    "📌 Always show VNINDEX as reference",
    value=True,
    help="VNINDEX will always appear at the bottom as a benchmark"
)

if not selected_types:
    st.warning("⚠️ Please select at least one asset type to analyze.")
    st.stop()

# Number of top performers
top_n = st.sidebar.slider(
    "Number of top performers per year",
    min_value=1,
    max_value=10,
    value=3,
    help="Select how many top performers to show for each year"
)

# Year range selector
st.sidebar.subheader("📅 Year Range")

# Calculate yearly returns first to get available years
with st.spinner("Loading available years..."):
    all_yearly_returns = calculate_yearly_returns()

if not all_yearly_returns.empty:
    available_years = sorted(all_yearly_returns['year'].unique())
    min_year = int(min(available_years))
    max_year = int(max(available_years))
    
    year_range = st.sidebar.slider(
        "Select year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        help="Filter the analysis to specific years"
    )
    
    # Filter yearly returns by year range
    yearly_returns = all_yearly_returns[
        (all_yearly_returns['year'] >= year_range[0]) & 
        (all_yearly_returns['year'] <= year_range[1])
    ].copy()
else:
    st.error("No data available. Please import data first.")
    st.stop()

# Get top performers
top_performers = get_top_performers(yearly_returns, selected_types, top_n=top_n)

if top_performers.empty:
    st.warning("No data available for selected asset types.")
    st.stop()

# Get VNINDEX data for reference
vnindex_data = yearly_returns[yearly_returns['asset_code'] == 'VNINDEX'].copy()

# Display summary statistics
st.subheader("📊 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Years Analyzed", len(top_performers['year'].unique()))

with col2:
    st.metric("Total Assets", len(top_performers['asset_code'].unique()))

with col3:
    avg_return = top_performers['yearly_return'].mean()
    st.metric("Avg Top Return", f"{avg_return:.1f}%")

with col4:
    max_return = top_performers['yearly_return'].max()
    max_asset = top_performers[top_performers['yearly_return'] == max_return].iloc[0]
    st.metric("Best Return", f"{max_return:.1f}%", 
              delta=f"{max_asset['asset_code']} ({int(max_asset['year'])})")

# Create table-style visualization similar to reference image
st.subheader("📈 Top Performers Timeline")

# Prepare data for table
years = sorted(top_performers['year'].unique())

# Debug: Show data info
with st.expander("🔍 Debug Info (click to expand)"):
    st.write(f"Total years: {len(years)}")
    st.write(f"Years: {years}")
    st.write(f"Top performers shape: {top_performers.shape}")
    st.write("Sample data:")
    st.dataframe(top_performers.head(10))

# Build HTML table using Streamlit components for better rendering
import streamlit.components.v1 as components

# Build HTML
html_content = """
<style>
.top-performers-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 8px;
    margin: 20px 0;
    font-family: Arial, sans-serif;
}

.top-performers-table th {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: white;
    padding: 12px 8px;
    text-align: center;
    font-weight: bold;
    font-size: 14px;
    border-radius: 8px;
    position: relative;
}

.top-performers-table th::before {
    content: '◆';
    position: absolute;
    top: -20px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 20px;
    color: #10B981;
}

.top-performers-table td {
    padding: 10px 8px;
    text-align: center;
    font-weight: bold;
    font-size: 11px;
    border-radius: 6px;
    color: white;
    min-width: 80px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.top-performers-table .outperform {
    border: 3px solid #FFD700 !important;
    box-shadow: 0 0 8px rgba(255, 215, 0, 0.5);
}

.top-performers-table .underperform {
    border: 2px solid #9CA3AF !important;
}

.top-performers-table .vnindex-cell {
    background: white;
    color: black;
    border: 2px solid black;
    font-style: normal;
}

.top-performers-table .asset-code {
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 2px;
}

.top-performers-table .return-value {
    font-size: 11px;
}
</style>

<table class="top-performers-table">
<thead><tr>
"""

# Add year headers
for year in years:
    html_content += f"<th>{int(year)}</th>"

html_content += "</tr></thead><tbody>"

# Helper function to get gradient color for US ETF
def get_gradient_color(base_color):
    """Generate a lighter gradient color from base color"""
    # Simple gradient by adjusting the hex color
    if base_color.startswith('#'):
        # Convert hex to RGB, lighten it, and return
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        
        # Lighten by 40% for more visible gradient
        r = min(255, int(r + (255 - r) * 0.4))
        g = min(255, int(g + (255 - g) * 0.4))
        b = min(255, int(b + (255 - b) * 0.4))
        
        return f"#{r:02x}{g:02x}{b:02x}"
    return base_color

# Add rows for each rank
for rank in range(1, top_n + 1):
    html_content += "<tr>"
    
    for year in years:
        year_data = top_performers[
            (top_performers['year'] == year) & 
            (top_performers['rank'] == rank)
        ]
        
        if not year_data.empty:
            row = year_data.iloc[0]
            asset_code = row['asset_code']
            asset_type = row['asset_type']
            yearly_return = row['yearly_return']
            color = get_asset_color(asset_code, rank)
            start_date = row['start_date'].strftime('%d/%m/%Y')
            end_date = row['end_date'].strftime('%d/%m/%Y')
            
            # Compare with VNINDEX for this year
            vnindex_year = vnindex_data[vnindex_data['year'] == year]
            border_class = ""
            
            if not vnindex_year.empty:
                vnindex_return = vnindex_year.iloc[0]['yearly_return']
                if yearly_return > vnindex_return:
                    border_class = "outperform"
                else:
                    border_class = "underperform"
            
            # Determine styling based on asset type
            if asset_type == 'us_etf':
                # US ETF: Gradient + Black border
                gradient_color = get_gradient_color(color)
                style = f"background: linear-gradient(135deg, {color} 0%, {gradient_color} 100%); border: 3px solid #000000;"
            elif asset_type == 'crypto':
                # Crypto: Solid color + Purple border
                style = f"background-color: {color}; border: 3px solid #8B5CF6;"
            elif asset_type == 'commodity':
                # Commodity: Solid color + Brown border
                style = f"background-color: {color}; border: 3px solid #92400E;"
            elif asset_type == 'us_stock':
                # US Stock: Solid color + Blue border
                style = f"background-color: {color}; border: 3px solid #1E40AF;"
            else:
                # VN Fund and others: Solid color, no special border (will use outperform/underperform border)
                style = f"background-color: {color};"
            
            # Add outperform/underperform border for VN funds only
            # For other asset types, the asset type border takes precedence
            if asset_type == 'vn_fund':
                html_content += f"""
                <td class="{border_class}" style="{style}" title="{asset_code} ({asset_type}): {yearly_return:.1f}% ({start_date} - {end_date})">
                    <div class="asset-code">{asset_code}</div>
                    <div class="return-value">{yearly_return:+.1f}%</div>
                </td>
                """
            else:
                # For non-VN funds, show asset type border instead
                html_content += f"""
                <td style="{style}" title="{asset_code} ({asset_type}): {yearly_return:.1f}% ({start_date} - {end_date})">
                    <div class="asset-code">{asset_code}</div>
                    <div class="return-value">{yearly_return:+.1f}%</div>
                </td>
                """
        else:
            html_content += '<td style="background-color: #f3f4f6;"></td>'
    
    html_content += "</tr>"

# Add VNINDEX row
if include_vnindex and not vnindex_data.empty:
    html_content += "<tr>"
    
    for year in years:
        vnindex_year = vnindex_data[vnindex_data['year'] == year]
        
        if not vnindex_year.empty:
            row = vnindex_year.iloc[0]
            yearly_return = row['yearly_return']
            start_date = row['start_date'].strftime('%d/%m/%Y')
            end_date = row['end_date'].strftime('%d/%m/%Y')
            
            html_content += f"""
            <td class="vnindex-cell" title="VNINDEX: {yearly_return:.1f}% ({start_date} - {end_date})">
                <div class="asset-code">VNINDEX</div>
                <div class="return-value">{yearly_return:+.1f}%</div>
            </td>
            """
        else:
            html_content += '<td class="vnindex-cell"></td>'
    
    html_content += "</tr>"

html_content += "</tbody></table>"

# Render using components
components.html(html_content, height=400, scrolling=True)

# Legend explanation
st.markdown("""
**📖 How to read:**
- **Gold border (✨)**: VN Fund outperformed VNINDEX
- **Black border**: US ETF
- **Purple border**: Crypto
- **Brown border**: Commodity  
- **Blue border**: US Stock
""")

# Detailed table
st.subheader("📋 Detailed Performance Table")

# Prepare table data
table_data = []

# Sort years in descending order (newest first)
for year in sorted(top_performers['year'].unique(), reverse=True):
    year_data = top_performers[top_performers['year'] == year].sort_values('rank')
    
    for idx, row in year_data.iterrows():
        table_data.append({
            'Year': int(row['year']),
            'Rank': f"#{int(row['rank'])}",
            'Asset': row['asset_code'],
            'Type': row['asset_type'],
            'Return': f"{row['yearly_return']:.2f}%",
            'Start Date': row['start_date'].strftime('%d/%m/%Y'),
            'End Date': row['end_date'].strftime('%d/%m/%Y'),
            'Start Price': f"{row['start_price']:.2f}",
            'End Price': f"{row['end_price']:.2f}"
        })
    
    # Add VNINDEX reference if available
    if include_vnindex and not vnindex_data.empty:
        vnindex_year = vnindex_data[vnindex_data['year'] == year]
        if not vnindex_year.empty:
            row = vnindex_year.iloc[0]
            table_data.append({
                'Year': int(row['year']),
                'Rank': 'REF',
                'Asset': 'VNINDEX',
                'Type': 'vn_index',
                'Return': f"{row['yearly_return']:.2f}%",
                'Start Date': row['start_date'].strftime('%d/%m/%Y'),
                'End Date': row['end_date'].strftime('%d/%m/%Y'),
                'Start Price': f"{row['start_price']:.2f}",
                'End Price': f"{row['end_price']:.2f}"
            })

table_df = pd.DataFrame(table_data)

# Style the table
def style_table(row):
    """Apply styling to table rows"""
    if row['Rank'] == '#1':
        return ['background-color: #fef3c7'] * len(row)  # Gold
    elif row['Rank'] == '#2':
        return ['background-color: #e5e7eb'] * len(row)  # Silver
    elif row['Rank'] == '#3':
        return ['background-color: #fed7aa'] * len(row)  # Bronze
    elif row['Rank'] == 'REF':
        return ['background-color: #f3f4f6; font-style: italic'] * len(row)  # Gray
    return [''] * len(row)

styled_table = table_df.style.apply(style_table, axis=1)

st.dataframe(styled_table, use_container_width=True, height=400)

# Download button
csv = table_df.to_csv(index=False)
st.download_button(
    label="📥 Download Table as CSV",
    data=csv,
    file_name=f"top_performers_{min(years)}_{max(years)}.csv",
    mime="text/csv"
)

# Asset performance summary
st.subheader("🎯 Asset Performance Summary")

# Count how many times each asset appears in top performers
asset_summary = top_performers.groupby('asset_code').agg({
    'year': 'count',
    'rank': ['mean', 'min'],
    'yearly_return': ['mean', 'max', 'min']
}).round(2)

asset_summary.columns = ['Appearances', 'Avg Rank', 'Best Rank', 'Avg Return (%)', 'Max Return (%)', 'Min Return (%)']
asset_summary = asset_summary.sort_values('Appearances', ascending=False)

st.dataframe(asset_summary, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>
    <p>💡 <b>Tip:</b> Use the sidebar filters to explore different asset types and adjust the number of top performers.</p>
    <p>Data is calculated based on calendar year returns (Jan 1 - Dec 31)</p>
</div>
""", unsafe_allow_html=True)
