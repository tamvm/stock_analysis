import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sqlite3
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Fund Holdings Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color palette for stocks (vibrant, distinguishable colors)
STOCK_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
    '#E63946', '#A8DADC', '#457B9D', '#F1FAEE', '#E76F51',
    '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#264653'
]

def get_stock_color(stock_code, all_stocks):
    """Get consistent color for a stock based on its position in the list"""
    try:
        idx = all_stocks.index(stock_code)
        return STOCK_COLORS[idx % len(STOCK_COLORS)]
    except ValueError:
        return STOCK_COLORS[0]

@st.cache_data
def get_available_funds(db_path="db/investment_data.db"):
    """Get list of funds that have holdings data"""
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT DISTINCT 
            fi.asset_code,
            fi.fund_short_name,
            fi.issuer_short_name,
            COUNT(fh.stock_code) as num_holdings
        FROM fund_info fi
        JOIN fund_holdings fh ON fi.asset_code = fh.asset_code
        GROUP BY fi.asset_code, fi.fund_short_name, fi.issuer_short_name
        ORDER BY fi.asset_code
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

@st.cache_data
def get_fund_holdings(asset_codes, db_path="db/investment_data.db"):
    """Get holdings data for selected funds"""
    conn = sqlite3.connect(db_path)
    
    placeholders = ','.join(['?' for _ in asset_codes])
    query = f"""
        SELECT 
            asset_code,
            stock_code,
            net_asset_percent,
            industry,
            price,
            price_change_percent
        FROM fund_holdings
        WHERE asset_code IN ({placeholders})
        ORDER BY asset_code, net_asset_percent DESC
    """
    
    df = pd.read_sql_query(query, conn, params=asset_codes)
    conn.close()
    
    return df

def calculate_overlap_metrics(holdings_df, fund_a, fund_b):
    """Calculate overlap metrics between two funds"""
    fund_a_stocks = set(holdings_df[holdings_df['asset_code'] == fund_a]['stock_code'].tolist())
    fund_b_stocks = set(holdings_df[holdings_df['asset_code'] == fund_b]['stock_code'].tolist())
    
    common_stocks = fund_a_stocks & fund_b_stocks
    
    # Calculate total weight of common stocks in each fund
    fund_a_common_weight = holdings_df[
        (holdings_df['asset_code'] == fund_a) & 
        (holdings_df['stock_code'].isin(common_stocks))
    ]['net_asset_percent'].sum()
    
    fund_b_common_weight = holdings_df[
        (holdings_df['asset_code'] == fund_b) & 
        (holdings_df['stock_code'].isin(common_stocks))
    ]['net_asset_percent'].sum()
    
    return {
        'common_count': len(common_stocks),
        'common_stocks': common_stocks,
        'fund_a_overlap_weight': fund_a_common_weight,
        'fund_b_overlap_weight': fund_b_common_weight
    }

# Title
st.title("📊 Fund Holdings Comparison")
st.markdown("**Compare top stock holdings between two Vietnamese funds**")

# Load available funds
available_funds = get_available_funds()

if available_funds.empty:
    st.error("No fund holdings data available. Please run `import_vn_fund_metadata.py` first.")
    st.stop()

# Sidebar - Fund Selection
st.sidebar.header("Fund Selection")
st.sidebar.markdown("*Select exactly 2 funds to compare*")

# Create fund options with display names
fund_options = {}
for _, row in available_funds.iterrows():
    display_name = f"{row['asset_code']} - {row['fund_short_name']}"
    if pd.notna(row['issuer_short_name']):
        display_name += f" ({row['issuer_short_name']})"
    fund_options[display_name] = row['asset_code']

# Get URL query parameters
query_params = st.query_params
url_funds = query_params.get('funds', None)

# Parse URL funds if present
default_selected = []
if url_funds:
    url_fund_codes = [f.strip() for f in url_funds.split(',')]
    # Match URL fund codes to display names
    for display_name, asset_code in fund_options.items():
        if asset_code in url_fund_codes:
            default_selected.append(display_name)

# Fund multiselect
selected_fund_names = st.sidebar.multiselect(
    "Select 2 funds:",
    options=list(fund_options.keys()),
    default=default_selected[:2] if len(default_selected) >= 2 else [],
    help="Choose exactly 2 funds to compare their holdings"
)

# Update URL parameters
if len(selected_fund_names) == 2:
    selected_codes = [fund_options[name] for name in selected_fund_names]
    st.query_params['funds'] = ','.join(selected_codes)

# Validation
if len(selected_fund_names) == 0:
    st.info("👆 Please select 2 funds from the sidebar to begin comparison.")
    st.stop()
elif len(selected_fund_names) == 1:
    st.warning("⚠️ Please select one more fund to compare.")
    st.stop()
elif len(selected_fund_names) > 2:
    st.error("❌ Please select exactly 2 funds. You have selected " + str(len(selected_fund_names)) + " funds.")
    st.stop()

# Get selected fund codes
fund_a_name = selected_fund_names[0]
fund_b_name = selected_fund_names[1]
fund_a = fund_options[fund_a_name]
fund_b = fund_options[fund_b_name]

# Load holdings data
holdings_df = get_fund_holdings([fund_a, fund_b])

if holdings_df.empty:
    st.error("No holdings data found for selected funds.")
    st.stop()

# Calculate overlap metrics
overlap = calculate_overlap_metrics(holdings_df, fund_a, fund_b)

# Display fund names
st.markdown(f"### Comparing: **{fund_a}** vs **{fund_b}**")

# Summary Cards
st.subheader("📈 Overlap Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Common Holdings",
        f"{overlap['common_count']} stocks",
        help="Number of stocks held by both funds"
    )

with col2:
    st.metric(
        f"{fund_a} Overlap Weight",
        f"{overlap['fund_a_overlap_weight']:.1f}%",
        help=f"Total portfolio weight of common stocks in {fund_a}"
    )

with col3:
    st.metric(
        f"{fund_b} Overlap Weight",
        f"{overlap['fund_b_overlap_weight']:.1f}%",
        help=f"Total portfolio weight of common stocks in {fund_b}"
    )

# Prepare data for visualization
# Get top 10 from each fund
fund_a_top10 = holdings_df[holdings_df['asset_code'] == fund_a].head(10)
fund_b_top10 = holdings_df[holdings_df['asset_code'] == fund_b].head(10)

# Get union of all stocks (could be 10-20 stocks)
all_stocks = list(dict.fromkeys(
    fund_a_top10['stock_code'].tolist() + fund_b_top10['stock_code'].tolist()
))

# Create comparison dataframe
comparison_data = []
for stock in all_stocks:
    fund_a_data = holdings_df[
        (holdings_df['asset_code'] == fund_a) & 
        (holdings_df['stock_code'] == stock)
    ]
    fund_b_data = holdings_df[
        (holdings_df['asset_code'] == fund_b) & 
        (holdings_df['stock_code'] == stock)
    ]
    
    comparison_data.append({
        'stock_code': stock,
        'fund_a_weight': fund_a_data['net_asset_percent'].values[0] if not fund_a_data.empty else 0,
        'fund_b_weight': fund_b_data['net_asset_percent'].values[0] if not fund_b_data.empty else 0,
        'fund_a_rank': list(fund_a_top10['stock_code']).index(stock) + 1 if stock in fund_a_top10['stock_code'].values else None,
        'fund_b_rank': list(fund_b_top10['stock_code']).index(stock) + 1 if stock in fund_b_top10['stock_code'].values else None,
        'is_common': stock in overlap['common_stocks'],
        'max_weight': max(
            fund_a_data['net_asset_percent'].values[0] if not fund_a_data.empty else 0,
            fund_b_data['net_asset_percent'].values[0] if not fund_b_data.empty else 0
        )
    })

comparison_df = pd.DataFrame(comparison_data)
# Sort by max weight descending
comparison_df = comparison_df.sort_values('max_weight', ascending=False)

# Grouped Bar Chart
st.subheader("📊 Holdings Comparison Chart")

fig = go.Figure()

# Add Fund A bars
fig.add_trace(go.Bar(
    name=fund_a,
    x=comparison_df['stock_code'],
    y=comparison_df['fund_a_weight'],
    marker_color='#4ECDC4',
    marker_line_color='#2C7A7B',
    marker_line_width=2,
    text=comparison_df['fund_a_weight'].apply(lambda x: f'{x:.1f}%' if x > 0 else ''),
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>' +
                  f'{fund_a}<br>' +
                  'Weight: %{y:.2f}%<br>' +
                  '<extra></extra>'
))

# Add Fund B bars
fig.add_trace(go.Bar(
    name=fund_b,
    x=comparison_df['stock_code'],
    y=comparison_df['fund_b_weight'],
    marker_color='#FF6B6B',
    marker_line_color='#C53030',
    marker_line_width=2,
    text=comparison_df['fund_b_weight'].apply(lambda x: f'{x:.1f}%' if x > 0 else ''),
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>' +
                  f'{fund_b}<br>' +
                  'Weight: %{y:.2f}%<br>' +
                  '<extra></extra>'
))

# Highlight common stocks with background color
for idx, row in comparison_df.iterrows():
    if row['is_common']:
        fig.add_vrect(
            x0=idx - 0.4,
            x1=idx + 0.4,
            fillcolor="rgba(255, 215, 0, 0.1)",
            layer="below",
            line_width=0,
        )

fig.update_layout(
    barmode='group',
    title=f"Top Holdings: {fund_a} vs {fund_b}",
    xaxis_title="Stock Code",
    yaxis_title="Portfolio Weight (%)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=500,
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

st.plotly_chart(fig, use_container_width=True)

# Add legend for common stocks
st.markdown("""
**📖 Chart Legend:**
- **Yellow background**: Common holdings (held by both funds)
- **Teal bars**: """ + fund_a + """
- **Red bars**: """ + fund_b + """
""")

# Comparison Table
st.subheader("📋 Detailed Comparison Table")

# Prepare table data
table_data = []
for _, row in comparison_df.iterrows():
    stock = row['stock_code']
    
    # Get industry info
    stock_info = holdings_df[holdings_df['stock_code'] == stock].iloc[0]
    
    table_data.append({
        'Stock Code': stock,
        'Industry': stock_info['industry'] if pd.notna(stock_info['industry']) else 'N/A',
        f'{fund_a} Weight (%)': f"{row['fund_a_weight']:.2f}" if row['fund_a_weight'] > 0 else 'N/A',
        f'{fund_a} Rank': f"#{int(row['fund_a_rank'])}" if pd.notna(row['fund_a_rank']) else 'N/A',
        f'{fund_b} Weight (%)': f"{row['fund_b_weight']:.2f}" if row['fund_b_weight'] > 0 else 'N/A',
        f'{fund_b} Rank': f"#{int(row['fund_b_rank'])}" if pd.notna(row['fund_b_rank']) else 'N/A',
        'Status': '✅ Common' if row['is_common'] else (
            f'📌 {fund_a} Only' if row['fund_a_weight'] > 0 else f'📌 {fund_b} Only'
        )
    })

table_df = pd.DataFrame(table_data)

# Style the table
def highlight_common(row):
    """Highlight common stocks in green"""
    if '✅ Common' in row['Status']:
        return ['background-color: #D1FAE5'] * len(row)
    return [''] * len(row)

styled_table = table_df.style.apply(highlight_common, axis=1)

st.dataframe(styled_table, use_container_width=True, height=400)

# Download button
csv = table_df.to_csv(index=False)
st.download_button(
    label="📥 Download Comparison as CSV",
    data=csv,
    file_name=f"fund_comparison_{fund_a}_vs_{fund_b}_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# Additional insights
st.subheader("💡 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**{fund_a} Top 3 Holdings:**")
    for idx, row in fund_a_top10.head(3).iterrows():
        is_common = row['stock_code'] in overlap['common_stocks']
        marker = "✅" if is_common else "📌"
        st.markdown(f"{marker} **{row['stock_code']}**: {row['net_asset_percent']:.2f}%")

with col2:
    st.markdown(f"**{fund_b} Top 3 Holdings:**")
    for idx, row in fund_b_top10.head(3).iterrows():
        is_common = row['stock_code'] in overlap['common_stocks']
        marker = "✅" if is_common else "📌"
        st.markdown(f"{marker} **{row['stock_code']}**: {row['net_asset_percent']:.2f}%")

# Sector Allocation Comparison
st.markdown("---")
st.subheader("🏭 Sector Allocation Comparison")
st.markdown("**Compare how each fund distributes investments across different industries**")

# Query sector allocation data
@st.cache_data
def get_sector_allocation(asset_codes, db_path="db/investment_data.db"):
    """Get sector allocation data for selected funds"""
    conn = sqlite3.connect(db_path)
    
    placeholders = ','.join(['?' for _ in asset_codes])
    query = f"""
        SELECT 
            asset_code,
            industry,
            asset_percent
        FROM fund_sector_allocation
        WHERE asset_code IN ({placeholders})
        ORDER BY asset_code, asset_percent DESC
    """
    
    df = pd.read_sql_query(query, conn, params=asset_codes)
    conn.close()
    
    return df

# Load sector allocation data
sector_df = get_sector_allocation([fund_a, fund_b])

if not sector_df.empty:
    # Get all unique sectors from both funds
    all_sectors = list(dict.fromkeys(
        sector_df['industry'].unique().tolist()
    ))
    
    # Create sector comparison dataframe
    sector_comparison = []
    for sector in all_sectors:
        fund_a_data = sector_df[
            (sector_df['asset_code'] == fund_a) & 
            (sector_df['industry'] == sector)
        ]
        fund_b_data = sector_df[
            (sector_df['asset_code'] == fund_b) & 
            (sector_df['industry'] == sector)
        ]
        
        sector_comparison.append({
            'sector': sector,
            'fund_a_weight': fund_a_data['asset_percent'].values[0] if not fund_a_data.empty else 0,
            'fund_b_weight': fund_b_data['asset_percent'].values[0] if not fund_b_data.empty else 0,
            'max_weight': max(
                fund_a_data['asset_percent'].values[0] if not fund_a_data.empty else 0,
                fund_b_data['asset_percent'].values[0] if not fund_b_data.empty else 0
            ),
            'in_both': not fund_a_data.empty and not fund_b_data.empty
        })
    
    sector_comp_df = pd.DataFrame(sector_comparison)
    # Sort by max weight descending
    sector_comp_df = sector_comp_df.sort_values('max_weight', ascending=False)
    
    # Sector Allocation Bar Chart
    fig_sector = go.Figure()
    
    # Add Fund A bars
    fig_sector.add_trace(go.Bar(
        name=fund_a,
        x=sector_comp_df['sector'],
        y=sector_comp_df['fund_a_weight'],
        marker_color='#4ECDC4',
        marker_line_color='#2C7A7B',
        marker_line_width=2,
        text=sector_comp_df['fund_a_weight'].apply(lambda x: f'{x:.1f}%' if x > 0 else ''),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                      f'{fund_a}<br>' +
                      'Weight: %{y:.2f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Add Fund B bars
    fig_sector.add_trace(go.Bar(
        name=fund_b,
        x=sector_comp_df['sector'],
        y=sector_comp_df['fund_b_weight'],
        marker_color='#FF6B6B',
        marker_line_color='#C53030',
        marker_line_width=2,
        text=sector_comp_df['fund_b_weight'].apply(lambda x: f'{x:.1f}%' if x > 0 else ''),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                      f'{fund_b}<br>' +
                      'Weight: %{y:.2f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Add Total bars (sum of both funds)
    sector_comp_df['total_weight'] = sector_comp_df['fund_a_weight'] + sector_comp_df['fund_b_weight']
    fig_sector.add_trace(go.Bar(
        name='Total',
        x=sector_comp_df['sector'],
        y=sector_comp_df['total_weight'],
        marker_color='#FCD34D',
        marker_line_color='#F59E0B',
        marker_line_width=2,
        text=sector_comp_df['total_weight'].apply(lambda x: f'{x:.1f}%' if x > 0 else ''),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                      'Total (Both Funds)<br>' +
                      'Weight: %{y:.2f}%<br>' +
                      '<extra></extra>'
    ))
    
    fig_sector.update_layout(
        barmode='group',
        title=f"Sector Allocation: {fund_a} vs {fund_b}",
        xaxis_title="Industry Sector",
        yaxis_title="Portfolio Weight (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig_sector.update_xaxes(showgrid=False, tickangle=-45)
    fig_sector.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig_sector, use_container_width=True)
    
    # Sector comparison table
    st.markdown("**📊 Sector Allocation Details**")
    
    sector_table_data = []
    for _, row in sector_comp_df.iterrows():
        diff = row['fund_a_weight'] - row['fund_b_weight']
        sector_table_data.append({
            'Sector': row['sector'],
            f'{fund_a} (%)': f"{row['fund_a_weight']:.2f}" if row['fund_a_weight'] > 0 else 'N/A',
            f'{fund_b} (%)': f"{row['fund_b_weight']:.2f}" if row['fund_b_weight'] > 0 else 'N/A',
            'Difference': diff,
            'Difference_Display': f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
        })
    
    sector_table_df = pd.DataFrame(sector_table_data)
    
    # Drop the numeric Difference column, keep only Display version
    display_df = sector_table_df.drop(columns=['Difference'])
    display_df = display_df.rename(columns={'Difference_Display': 'Difference'})
    
    # Style the sector table with color-coded difference
    def style_difference(row):
        """Color code the difference column"""
        styles = [''] * len(row)
        
        # Find the Difference column index
        try:
            diff_idx = list(row.index).index('Difference')
            # Get the original numeric difference from sector_table_df
            sector_name = row['Sector']
            original_diff = sector_table_df[sector_table_df['Sector'] == sector_name]['Difference'].values[0]
            
            if original_diff > 0:
                styles[diff_idx] = 'color: #16A34A; font-weight: 600'  # Green for Fund A higher
            elif original_diff < 0:
                styles[diff_idx] = 'color: #DC2626; font-weight: 600'  # Red for Fund B higher
            else:
                styles[diff_idx] = 'color: #6B7280'  # Gray for equal
        except (ValueError, IndexError):
            pass
        
        return styles
    
    styled_sector_table = display_df.style.apply(style_difference, axis=1)
    
    st.dataframe(styled_sector_table, use_container_width=True, height=300)
    
    # Sector insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{fund_a} Top 3 Sectors:**")
        fund_a_sectors = sector_df[sector_df['asset_code'] == fund_a].sort_values('asset_percent', ascending=False).head(3)
        for _, row in fund_a_sectors.iterrows():
            st.markdown(f"• **{row['industry']}**: {row['asset_percent']:.2f}%")
    
    with col2:
        st.markdown(f"**{fund_b} Top 3 Sectors:**")
        fund_b_sectors = sector_df[sector_df['asset_code'] == fund_b].sort_values('asset_percent', ascending=False).head(3)
        for _, row in fund_b_sectors.iterrows():
            st.markdown(f"• **{row['industry']}**: {row['asset_percent']:.2f}%")
else:
    st.info("No sector allocation data available for selected funds.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>
    <p>💡 <b>Tip:</b> Common holdings indicate similar investment strategies between funds.</p>
    <p>Data shows the latest snapshot of fund holdings (top 10 per fund)</p>
</div>
""", unsafe_allow_html=True)
