import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from data_processor import DataProcessor
from metrics_calculator import MetricsCalculator
from config import ASSET_PRESETS, ROLLING_WINDOW_DAYS, RISK_FREE_RATE

# Page config
st.set_page_config(
    page_title="Investment Analysis Portal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define better color palette for charts
CHART_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#1f77b4',  # Repeat colors if more assets
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd'
]

# Initialize processors
@st.cache_resource
def init_processors():
    return DataProcessor(), MetricsCalculator()

processor, calc = init_processors()

# Initialize localStorage persistence functions
def save_settings_to_localStorage():
    """Save current settings to browser localStorage"""
    settings = {
        'selected_assets': st.session_state.get('selected_assets_ls', []),
        'analysis_periods': st.session_state.get('analysis_periods_ls', ['1Y', '3Y', '5Y']),
        'rolling_years': st.session_state.get('rolling_years_ls', 4),
        'start_date': st.session_state.get('start_date_ls', ''),
        'end_date': st.session_state.get('end_date_ls', '')
    }

    # Convert date objects to strings for JSON serialization
    if isinstance(settings['start_date'], datetime):
        settings['start_date'] = settings['start_date'].strftime('%d/%m/%Y')
    if isinstance(settings['end_date'], datetime):
        settings['end_date'] = settings['end_date'].strftime('%d/%m/%Y')

    st.markdown(f"""
    <script>
    try {{
        localStorage.setItem('investment_dashboard_settings', JSON.stringify({settings}));
        console.log('Settings saved to localStorage:', {settings});
    }} catch (e) {{
        console.error('Error saving to localStorage:', e);
    }}
    </script>
    """, unsafe_allow_html=True)

def load_settings_from_localStorage():
    """Load settings from browser localStorage"""
    # Initialize with default values if localStorage is empty
    default_settings = {
        'selected_assets': [],
        'analysis_periods': ['1Y', '3Y', '5Y'],
        'rolling_years': 4,
        'start_date': '',
        'end_date': ''
    }

    # Try to get settings from localStorage via JavaScript and session state
    if 'localStorage_loaded' not in st.session_state:
        st.session_state.localStorage_loaded = False

    return default_settings

# Load settings on app start
loaded_settings = load_settings_from_localStorage()

# Title
st.title("📈 Investment Analysis Portal")
st.markdown("Professional investment analysis and comparison tool")

# Add localStorage JavaScript handler
st.markdown("""
<script>
// Load settings from localStorage on page load
function loadSettingsFromLocalStorage() {
    try {
        const savedSettings = localStorage.getItem('investment_dashboard_settings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            console.log('Loaded settings from localStorage:', settings);

            // Send settings to Streamlit via session state
            window.parent.postMessage({
                type: 'localStorage_settings',
                data: settings
            }, '*');

            return settings;
        }
    } catch (e) {
        console.error('Error loading from localStorage:', e);
    }
    return null;
}

// Save settings to localStorage
function saveSettingsToLocalStorage(settings) {
    try {
        localStorage.setItem('investment_dashboard_settings', JSON.stringify(settings));
        console.log('Settings saved to localStorage:', settings);
    } catch (e) {
        console.error('Error saving to localStorage:', e);
    }
}

// Load settings when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadSettingsFromLocalStorage();
});

// Also try loading immediately
loadSettingsFromLocalStorage();
</script>
""", unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("Asset Selection & Settings")

# Load available assets
try:
    assets_df = processor.get_asset_list()
    # VN funds
    fund_assets = assets_df[assets_df['asset_type'] == 'vn_fund']['asset_code'].tolist()
    # US ETFs
    etf_assets = assets_df[assets_df['asset_type'] == 'us_etf']['asset_code'].tolist()
    # Benchmarks include: benchmark type, vn_index type
    benchmark_assets = assets_df[assets_df['asset_type'].isin(['benchmark', 'vn_index'])]['asset_code'].tolist()
    # Also get other asset types
    stock_assets = assets_df[assets_df['asset_type'] == 'us_stock']['asset_code'].tolist()
    crypto_assets = assets_df[assets_df['asset_type'] == 'crypto']['asset_code'].tolist()
    
    all_assets = fund_assets + etf_assets + benchmark_assets + stock_assets + crypto_assets
except:
    st.error("Database not found. Please run data_processor.py first to load data.")
    st.stop()

# Handle localStorage settings for asset selection
if 'saved_selected_assets' not in st.session_state:
    # Check if we have saved assets in localStorage that are valid
    st.session_state.saved_selected_assets = loaded_settings.get('selected_assets', [])

# Filter saved assets to only include valid ones
valid_saved_assets = [asset for asset in st.session_state.saved_selected_assets if asset in all_assets]

# Initialize preset selection state - this will hold the currently selected preset assets
if 'preset_selected_assets' not in st.session_state:
    st.session_state.preset_selected_assets = None

# Asset Presets Section (BEFORE multiselect)
st.sidebar.subheader("📌 Quick Presets")
st.sidebar.markdown("*Click to quickly select asset groups*")

# Create asset type mapping for resolving @type: specifications
asset_type_map = {
    'vn_fund': fund_assets,
    'us_etf': etf_assets,
    'us_stock': stock_assets,
    'crypto': crypto_assets,
    'benchmark': assets_df[assets_df['asset_type'] == 'benchmark']['asset_code'].tolist(),
    'vn_index': assets_df[assets_df['asset_type'] == 'vn_index']['asset_code'].tolist()
}

# Function to resolve preset assets (handles both explicit assets and @type: specifications)
def resolve_preset_assets(preset_definition):
    """
    Resolve a preset definition that may contain:
    - Explicit asset codes: 'DCDS', 'VOO', etc.
    - Asset type specifications: '@type:vn_fund', '@type:crypto', etc.
    
    Returns a list of actual asset codes.
    """
    resolved_assets = []
    for item in preset_definition:
        if isinstance(item, str) and item.startswith('@type:'):
            # Extract the asset type name
            asset_type = item[6:]  # Remove '@type:' prefix
            # Add all assets of this type
            if asset_type in asset_type_map:
                resolved_assets.extend(asset_type_map[asset_type])
        else:
            # Regular asset code
            resolved_assets.append(item)
    return resolved_assets

# Populate presets by resolving @type: specifications
presets = {}
for preset_name, preset_definition in ASSET_PRESETS.items():
    presets[preset_name] = resolve_preset_assets(preset_definition)

# Create preset buttons in a grid
preset_names = list(presets.keys())
cols_per_row = 2
rows = (len(preset_names) + cols_per_row - 1) // cols_per_row

for row in range(rows):
    cols = st.sidebar.columns(cols_per_row)
    for col_idx in range(cols_per_row):
        preset_idx = row * cols_per_row + col_idx
        if preset_idx < len(preset_names):
            preset_name = preset_names[preset_idx]
            preset_assets = presets[preset_name]
            
            # Filter to only valid assets
            valid_preset_assets = [a for a in preset_assets if a in all_assets]
            
            with cols[col_idx]:
                if st.button(
                    preset_name,
                    key=f"preset_{preset_name}",
                    help=f"Select: {', '.join(valid_preset_assets) if valid_preset_assets else 'No assets'}",
                    use_container_width=True,
                    disabled=len(valid_preset_assets) == 0
                ):
                    # Store the preset selection in session state
                    st.session_state.preset_selected_assets = valid_preset_assets
                    # Also update our tracking of saved assets
                    st.session_state.saved_selected_assets = valid_preset_assets

# Determine the default value for multiselect
# Priority: 1) Preset selection, 2) Saved assets, 3) VN Top Funds preset
if st.session_state.preset_selected_assets is not None:
    default_assets = st.session_state.preset_selected_assets
    # Clear the preset selection after using it
    st.session_state.preset_selected_assets = None
elif valid_saved_assets:
    default_assets = valid_saved_assets
else:
    # Use VN Top Funds as default
    vn_top_funds_preset = resolve_preset_assets(ASSET_PRESETS.get('VN Top Funds', []))
    default_assets = [a for a in vn_top_funds_preset if a in all_assets]
    # Fallback to first 3 funds if VN Top Funds preset is empty
    if not default_assets:
        default_assets = fund_assets[:3] if len(fund_assets) >= 3 else fund_assets

# Asset selection
st.sidebar.subheader("Select Assets to Compare")
selected_assets = st.sidebar.multiselect(
    "Choose Assets:",
    options=all_assets,
    default=default_assets,
    help="Select multiple assets to compare. Includes funds, ETFs, and benchmarks.",
    key="selected_assets_multiselect"
)

# Update session state for localStorage saving AND internal tracking
st.session_state.selected_assets_ls = selected_assets
st.session_state.saved_selected_assets = selected_assets

if not selected_assets:
    st.warning("Please select at least one asset to analyze.")
    st.stop()

# Asset detail navigation
st.sidebar.subheader("Asset Detail Analysis")
detail_asset = st.sidebar.selectbox(
    "View detailed analysis:",
    options=["Select asset..."] + all_assets,
    help="Select an asset to view detailed information and analysis"
)

if detail_asset != "Select asset...":
    if st.sidebar.button(f"📊 Analyze {detail_asset}", use_container_width=True):
        # Set the selected asset in session state and navigate
        st.session_state['selected_detail_asset'] = detail_asset
        st.switch_page("pages/2_📊_Asset_Detail.py")

# Advanced filtering navigation
st.sidebar.subheader("Advanced Analysis")
if st.sidebar.button("🔍 Advanced Filter & Analysis", use_container_width=True):
    st.switch_page("pages/3_🔍_Advanced_Filter.py")

# Assets list navigation
if st.sidebar.button("📋 View All Assets", use_container_width=True):
    st.switch_page("pages/4_📋_Assets_List.py")


# Handle localStorage settings for rolling years
saved_rolling_years = loaded_settings.get('rolling_years', 4)
rolling_years_options = [1, 2, 3, 4, 5]

# Find the index of saved value, default to 2 (4 years) if not found
try:
    rolling_years_index = rolling_years_options.index(saved_rolling_years)
except ValueError:
    rolling_years_index = 2  # Default to 4 years

# Rolling period selector
st.sidebar.subheader("Rolling CAGR Settings")
rolling_years = st.sidebar.selectbox(
    "Rolling CAGR Period:",
    options=rolling_years_options,
    index=rolling_years_index,
    help="Select the rolling period for CAGR calculation",
    key="rolling_years_selectbox"
)

# Update session state for localStorage saving
st.session_state.rolling_years_ls = rolling_years

# Handle localStorage settings for analysis periods
saved_analysis_periods = loaded_settings.get('analysis_periods', ['1Y', '3Y', '5Y'])
analysis_options = ['1Y', '3Y', '5Y', 'Inception']

# Filter saved periods to only include valid ones
valid_saved_periods = [period for period in saved_analysis_periods if period in analysis_options]

# Performance period selector
st.sidebar.subheader("Performance Analysis")
analysis_periods = st.sidebar.multiselect(
    "Analysis Periods:",
    options=analysis_options,
    default=valid_saved_periods if valid_saved_periods else ['1Y', '3Y', '5Y'],
    help="Select time periods for performance comparison",
    key="analysis_periods_multiselect"
)

# Update session state for localStorage saving
st.session_state.analysis_periods_ls = analysis_periods

# Date range selector
st.sidebar.subheader("Date Range (Optional)")

# Add buttons to set inception ranges
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("📅 Max Inception", help="Set date range from earliest fund inception to latest data"):
        try:
            # Get date range directly from price data for selected assets
            price_range_data = processor.get_price_data(selected_assets)

            if not price_range_data.empty:
                earliest_date = price_range_data['date'].min().date()
                latest_date = price_range_data['date'].max().date()

                # Update widget state directly
                st.session_state.start_date = earliest_date
                st.session_state.end_date = latest_date
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error setting max inception range: {e}")

with col2:
    if st.button("📅 Min Inception", help="Set date range from latest fund inception to latest data"):
        try:
            # Get date range directly from price data for selected assets
            price_range_data = processor.get_price_data(selected_assets)

            if not price_range_data.empty:
                # Group by asset_code and get the first date for each asset
                asset_start_dates = price_range_data.groupby('asset_code')['date'].min()

                # Find the latest inception date (minimum common period)
                latest_inception_date = asset_start_dates.max().date()
                latest_date = price_range_data['date'].max().date()

                # Update widget state directly
                st.session_state.start_date = latest_inception_date
                st.session_state.end_date = latest_date
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error setting min inception range: {e}")

# Handle localStorage settings for date range
saved_start_date = loaded_settings.get('start_date', '')
saved_end_date = loaded_settings.get('end_date', '')

# Convert saved date strings back to date objects if they exist and are valid
start_date_value = None
end_date_value = None

if saved_start_date:
    try:
        start_date_value = datetime.strptime(saved_start_date, '%d/%m/%Y').date()
    except (ValueError, TypeError):
        start_date_value = None

if saved_end_date:
    try:
        end_date_value = datetime.strptime(saved_end_date, '%d/%m/%Y').date()
    except (ValueError, TypeError):
        end_date_value = None

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=start_date_value,
        key="start_date"
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=end_date_value,
        key="end_date"
    )

# Update session state for localStorage saving
st.session_state.start_date_ls = start_date
st.session_state.end_date_ls = end_date

# Load data for selected assets
@st.cache_data
def load_selected_data(asset_codes, start_dt=None, end_dt=None):
    try:
        price_data, returns_data = calc.get_returns_data(asset_codes)

        if start_dt:
            price_data = price_data[price_data.index >= pd.to_datetime(start_dt)]
            returns_data = returns_data[returns_data.index >= pd.to_datetime(start_dt)]
        if end_dt:
            price_data = price_data[price_data.index <= pd.to_datetime(end_dt)]
            returns_data = returns_data[returns_data.index <= pd.to_datetime(end_dt)]

        return price_data, returns_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

price_data, returns_data = load_selected_data(selected_assets, start_date, end_date)

if price_data is None:
    st.stop()

# Performance Summary (First Row)
st.subheader("📊 Performance Summary")

try:
    summary = calc.get_performance_summary(selected_assets, analysis_periods if analysis_periods else ['3Y'])

    # Create summary table
    summary_data = []
    for asset in selected_assets:
        row = {'Asset': asset}

        for period in (analysis_periods if analysis_periods else ['3Y']):
            if period in summary:
                ann_ret = summary[period]['annualized_return'].get(asset, 'N/A')

                if isinstance(ann_ret, (int, float, np.floating)):
                    row[f'{period} Return'] = f"{ann_ret:.1f}%"
                else:
                    row[f'{period} Return'] = 'N/A'

        # Add overall metrics
        max_dd = summary['max_drawdown'].get(asset, 'N/A')

        # Use Sharpe ratio from the first analysis period
        first_period = analysis_periods[0] if analysis_periods else '3Y'
        sharpe = 'N/A'
        if first_period in summary and 'sharpe_ratio' in summary[first_period]:
            sharpe = summary[first_period]['sharpe_ratio'].get(asset, 'N/A')

        if isinstance(max_dd, (int, float, np.floating)):
            row['Max DD'] = f"{max_dd:.1f}%"
        else:
            row['Max DD'] = 'N/A'

        if isinstance(sharpe, (int, float, np.floating)):
            row['Sharpe'] = f"{sharpe:.2f}"
        else:
            row['Sharpe'] = 'N/A'

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    # Quick links to asset details
    st.markdown("**🔍 Quick Asset Analysis:**")
    cols = st.columns(min(len(selected_assets), 5))  # Max 5 columns

    for i, asset in enumerate(selected_assets):
        col_idx = i % 5
        with cols[col_idx]:
            if st.button(f"📊 {asset}", key=f"detail_btn_{asset}", help=f"View detailed analysis for {asset}"):
                st.session_state['selected_detail_asset'] = asset
                st.switch_page("pages/2_📊_Asset_Detail.py")

except Exception as e:
    st.error(f"Error calculating summary: {e}")

# Rolling CAGR Chart (Second Row)
st.subheader(f"📈 Rolling {rolling_years}-Year CAGR")

try:
    rolling_cagr = calc.calculate_rolling_cagr(price_data, years=rolling_years)

    if not rolling_cagr.empty:
        fig_cagr = go.Figure()

        for i, asset in enumerate(rolling_cagr.columns):
            data = rolling_cagr[asset].dropna()
            if len(data) > 0:
                # Create custom hover text with From/To dates
                hover_text = []
                for date, cagr_value in zip(data.index, data.values):
                    start_date = date - timedelta(days=rolling_years*365)
                    hover_text.append(
                        f"<b>{asset}</b><br>" +
                        f"From: {start_date.strftime('%d/%m/%Y')} To: {date.strftime('%d/%m/%Y')}<br>" +
                        f"CAGR: {cagr_value:.1f}%"
                    )

                fig_cagr.add_trace(go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines',
                    name=asset,
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=3),
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hover_text
                ))

        fig_cagr.update_layout(
            title=f"Rolling {rolling_years}-Year CAGR Comparison",
            xaxis_title="Date",
            yaxis_title="CAGR (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )

        fig_cagr.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        st.plotly_chart(fig_cagr, use_container_width=True)
    else:
        st.warning("Not enough data for rolling CAGR calculation.")

except Exception as e:
    st.error(f"Error creating rolling CAGR chart: {e}")

# Rolling CAGR Statistics (immediately after the chart)
try:
    if not rolling_cagr.empty:
        st.subheader(f"📊 Rolling {rolling_years}-Year CAGR Statistics")

        # Create three columns for the metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🏆 Best/Worst Rate**")
            best_worst_stats = []

            # For each time period, find which asset had the best and worst CAGR
            rolling_cagr_clean = rolling_cagr.dropna(how='all')

            for asset in rolling_cagr.columns:
                best_count = 0
                worst_count = 0
                total_comparisons = 0

                # Go through each time period
                for date in rolling_cagr_clean.index:
                    period_data = rolling_cagr_clean.loc[date].dropna()

                    # Only count periods where this asset has data and there are at least 2 assets to compare
                    if asset in period_data.index and len(period_data) >= 2:
                        total_comparisons += 1
                        asset_value = period_data[asset]

                        # Check if this asset has the highest CAGR in this period
                        if asset_value == period_data.max():
                            # Handle ties: if multiple assets have the same max value, each gets credit
                            max_count = (period_data == period_data.max()).sum()
                            best_count += 1 / max_count

                        # Check if this asset has the lowest CAGR in this period
                        if asset_value == period_data.min():
                            # Handle ties: if multiple assets have the same min value, each gets blame
                            min_count = (period_data == period_data.min()).sum()
                            worst_count += 1 / min_count

                if total_comparisons > 0:
                    best_rate = (best_count / total_comparisons) * 100
                    worst_rate = (worst_count / total_comparisons) * 100

                    best_worst_stats.append({
                        'Asset': asset,
                        'Best Rate': f"{best_rate:.1f}%",
                        'Worst Rate': f"{worst_rate:.1f}%",
                        'Periods': total_comparisons
                    })

            if best_worst_stats:
                best_worst_df = pd.DataFrame(best_worst_stats)
                st.dataframe(best_worst_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**📈 Rolling CAGR Summary**")
            total_stats = []
            for asset in rolling_cagr.columns:
                data = rolling_cagr[asset].dropna()
                if len(data) > 0:
                    min_cagr = data.min()
                    max_cagr = data.max()
                    total_cagr = data.sum()  # Sum of all rolling CAGR periods
                    range_cagr = max_cagr - min_cagr  # Difference between max and min
                    total_stats.append({
                        'Asset': asset,
                        'Min CAGR': f"{min_cagr:.1f}%",
                        'Max CAGR': f"{max_cagr:.1f}%",
                        'Total': f"{total_cagr:.1f}%",
                        'Range': f"{range_cagr:.1f}%"
                    })

            if total_stats:
                total_df = pd.DataFrame(total_stats)
                st.dataframe(total_df, use_container_width=True, hide_index=True)

        with col3:
            st.markdown("**📊 Central Tendency**")
            central_stats = []
            for asset in rolling_cagr.columns:
                data = rolling_cagr[asset].dropna()
                if len(data) > 0:
                    median_cagr = data.median()
                    mean_cagr = data.mean()
                    central_stats.append({
                        'Asset': asset,
                        'Median CAGR': f"{median_cagr:.1f}%",
                        'Average CAGR': f"{mean_cagr:.1f}%",
                        'Std Dev': f"{data.std():.1f}%"
                    })

            if central_stats:
                central_df = pd.DataFrame(central_stats)
                st.dataframe(central_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error calculating rolling CAGR statistics: {e}")

# Additional charts row
st.subheader("📊 Additional Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Cumulative Returns", "Price Changes", "Risk-Return", "Correlation", "Drawdown"])

with tab1:
    # Cumulative Returns Chart
    st.markdown("**📈 Chart Explanation**: This chart shows how $100 invested in each asset would have grown over time. All assets start at 100% and the lines show relative performance.")

    try:
        # Check if we have price data
        if price_data.empty:
            st.warning("No price data available for selected assets.")
        else:
            # Normalize each asset to its own first valid value (100% start)
            cumulative_returns = pd.DataFrame(index=price_data.index)

            for asset in price_data.columns:
                asset_prices = price_data[asset].dropna()
                if not asset_prices.empty:
                    # Normalize to 100% from the first valid price of this asset
                    normalized_series = (asset_prices / asset_prices.iloc[0] * 100)
                    cumulative_returns[asset] = normalized_series

            if cumulative_returns.empty:
                st.warning("No cumulative returns data available after processing.")
            else:
                fig_cum = go.Figure()
                traces_added = 0

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
                        traces_added += 1

                if traces_added > 0:
                    fig_cum.update_layout(
                        title="Cumulative Returns (Normalized to 100%)",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Value (%)",
                        hovermode="x unified",
                        height=400
                    )

                    fig_cum.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)

                    st.plotly_chart(fig_cum, use_container_width=True)
                else:
                    st.warning("No valid data traces could be created for the cumulative returns chart.")


    except Exception as e:
        st.error(f"Error creating cumulative returns chart: {e}")

with tab2:
    # Price Changes Chart
    try:
        if price_data.empty:
            st.warning("No price data available for selected assets.")
        else:
            # Create subplots - one for absolute prices, one for percentage changes
            fig_price_changes = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Absolute Prices", "Daily Percentage Changes"),
                vertical_spacing=0.08,
                shared_xaxes=True
            )

            # Top subplot: Absolute prices
            for i, asset in enumerate(price_data.columns):
                data = price_data[asset].dropna()
                if len(data) > 0:
                    fig_price_changes.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data.values,
                            mode='lines',
                            name=f"{asset} (Price)",
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                            hovertemplate=f'<b>{asset}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Price: %{y:.2f}<br>' +
                                        '<extra></extra>',
                            showlegend=False
                        ),
                        row=1, col=1
                    )

            # Bottom subplot: Daily percentage changes
            for i, asset in enumerate(price_data.columns):
                data = price_data[asset].dropna()
                if len(data) > 1:
                    # Calculate daily percentage changes
                    pct_changes = data.pct_change() * 100
                    pct_changes = pct_changes.dropna()

                    if len(pct_changes) > 0:
                        fig_price_changes.add_trace(
                            go.Scatter(
                                x=pct_changes.index,
                                y=pct_changes.values,
                                mode='lines',
                                name=asset,
                                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                                hovertemplate=f'<b>{asset}</b><br>' +
                                            'Date: %{x}<br>' +
                                            'Daily Change: %{y:.2f}%<br>' +
                                            '<extra></extra>'
                            ),
                            row=2, col=1
                        )

            # Update layout
            fig_price_changes.update_layout(
                title="Price Changes Analysis",
                height=700,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )

            # Update y-axes labels
            fig_price_changes.update_yaxes(title_text="Price", row=1, col=1)
            fig_price_changes.update_yaxes(title_text="Daily Change (%)", row=2, col=1)
            fig_price_changes.update_xaxes(title_text="Date", row=2, col=1)

            # Add horizontal line at 0% for percentage changes
            fig_price_changes.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

            st.plotly_chart(fig_price_changes, use_container_width=True)

            # Add summary statistics table
            st.subheader("Price Change Statistics")

            price_stats = []
            for asset in price_data.columns:
                data = price_data[asset].dropna()
                if len(data) > 1:
                    # Calculate statistics
                    start_price = data.iloc[0]
                    end_price = data.iloc[-1]
                    total_change = ((end_price - start_price) / start_price) * 100

                    daily_changes = data.pct_change().dropna() * 100
                    avg_daily_change = daily_changes.mean()
                    volatility = daily_changes.std()
                    max_daily_gain = daily_changes.max()
                    max_daily_loss = daily_changes.min()

                    price_stats.append({
                        'Asset': asset,
                        'Start Price': f"{start_price:.2f}",
                        'End Price': f"{end_price:.2f}",
                        'Total Change': f"{total_change:.1f}%",
                        'Avg Daily Change': f"{avg_daily_change:.2f}%",
                        'Daily Volatility': f"{volatility:.2f}%",
                        'Max Daily Gain': f"{max_daily_gain:.2f}%",
                        'Max Daily Loss': f"{max_daily_loss:.2f}%"
                    })

            if price_stats:
                stats_df = pd.DataFrame(price_stats)
                st.dataframe(stats_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating price changes chart: {e}")

with tab3:
    # Risk-Return Analysis with Rolling Metrics
    st.markdown("**📊 Risk-Return Analysis**: Compare risk-adjusted performance across assets")
    
    # Risk-Return Scatter Plot
    try:
        if analysis_periods:
            period = analysis_periods[0]  # Use first selected period

            if period in summary:
                risk_return_data = []

                for asset in selected_assets:
                    ret = summary[period]['annualized_return'].get(asset)
                    vol = summary[period]['volatility'].get(asset)

                    if isinstance(ret, (int, float)) and isinstance(vol, (int, float)):
                        risk_return_data.append({
                            'Asset': asset,
                            'Return': ret,
                            'Risk': vol
                        })

                if risk_return_data:
                    rr_df = pd.DataFrame(risk_return_data)

                    fig_rr = px.scatter(
                        rr_df,
                        x='Risk',
                        y='Return',
                        text='Asset',
                        title=f"Risk vs Return ({period})",
                        labels={'Risk': 'Volatility (%)', 'Return': 'Annualized Return (%)'}
                    )

                    fig_rr.update_traces(textposition='top center', marker_size=10)
                    fig_rr.update_layout(height=400)

                    st.plotly_chart(fig_rr, use_container_width=True)
                else:
                    st.warning("Not enough data for risk-return analysis.")
    except Exception as e:
        st.error(f"Error creating risk-return chart: {e}")
    
    # Rolling Standard Deviation (Volatility)
    st.markdown(f"### Rolling {ROLLING_WINDOW_DAYS}-Day Standard Deviation (Volatility)")
    st.markdown(f"*Shows how risk changes over time. Higher values indicate higher volatility.*")
    
    try:
        if not returns_data.empty:
            # Calculate rolling standard deviation
            rolling_std = returns_data.rolling(window=ROLLING_WINDOW_DAYS).std() * np.sqrt(252) * 100  # Annualized
            
            if not rolling_std.empty:
                fig_std = go.Figure()
                
                for i, asset in enumerate(rolling_std.columns):
                    data = rolling_std[asset].dropna()
                    if len(data) > 0:
                        fig_std.add_trace(go.Scatter(
                            x=data.index,
                            y=data.values,
                            mode='lines',
                            name=asset,
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                            hovertemplate=f'<b>{asset}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Volatility: %{y:.1f}%<br>' +
                                        '<extra></extra>'
                        ))
                
                fig_std.update_layout(
                    title=f"Rolling {ROLLING_WINDOW_DAYS}-Day Annualized Standard Deviation",
                    xaxis_title="Date",
                    yaxis_title="Annualized Volatility (%)",
                    hovermode="x unified",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_std, use_container_width=True)
                
                # Volatility statistics
                st.markdown("#### Volatility Statistics")
                vol_stats = []
                for asset in rolling_std.columns:
                    data = rolling_std[asset].dropna()
                    if len(data) > 0:
                        vol_stats.append({
                            'Asset': asset,
                            'Current Vol': f"{data.iloc[-1]:.1f}%",
                            'Avg Vol': f"{data.mean():.1f}%",
                            'Min Vol': f"{data.min():.1f}%",
                            'Max Vol': f"{data.max():.1f}%"
                        })
                
                if vol_stats:
                    vol_df = pd.DataFrame(vol_stats)
                    st.dataframe(vol_df, use_container_width=True, hide_index=True)
            else:
                st.warning(f"Not enough data for {ROLLING_WINDOW_DAYS}-day rolling volatility calculation.")
    except Exception as e:
        st.error(f"Error creating rolling volatility chart: {e}")
    
    # Rolling Sharpe Ratio
    st.markdown(f"### Rolling {ROLLING_WINDOW_DAYS}-Day Sharpe Ratio")
    st.markdown(f"*Risk-adjusted returns using {RISK_FREE_RATE*100:.1f}% risk-free rate. Higher is better.*")
    
    try:
        if not returns_data.empty:
            # Calculate rolling Sharpe ratio
            rolling_mean = returns_data.rolling(window=ROLLING_WINDOW_DAYS).mean() * 252  # Annualized return
            rolling_std = returns_data.rolling(window=ROLLING_WINDOW_DAYS).std() * np.sqrt(252)  # Annualized std
            rolling_sharpe = (rolling_mean - RISK_FREE_RATE) / rolling_std
            
            if not rolling_sharpe.empty:
                fig_sharpe = go.Figure()
                
                for i, asset in enumerate(rolling_sharpe.columns):
                    data = rolling_sharpe[asset].dropna()
                    if len(data) > 0:
                        fig_sharpe.add_trace(go.Scatter(
                            x=data.index,
                            y=data.values,
                            mode='lines',
                            name=asset,
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                            hovertemplate=f'<b>{asset}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Sharpe Ratio: %{y:.2f}<br>' +
                                        '<extra></extra>'
                        ))
                
                fig_sharpe.update_layout(
                    title=f"Rolling {ROLLING_WINDOW_DAYS}-Day Sharpe Ratio",
                    xaxis_title="Date",
                    yaxis_title="Sharpe Ratio",
                    hovermode="x unified",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Add reference lines
                fig_sharpe.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_sharpe.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.3, 
                                    annotation_text="Good (>1)", annotation_position="right")
                
                st.plotly_chart(fig_sharpe, use_container_width=True)
                
                # Sharpe ratio statistics
                st.markdown("#### Sharpe Ratio Statistics")
                sharpe_stats = []
                for asset in rolling_sharpe.columns:
                    data = rolling_sharpe[asset].dropna()
                    if len(data) > 0:
                        # Count periods where Sharpe > 1 (good performance)
                        good_periods = (data > 1).sum()
                        total_periods = len(data)
                        good_rate = (good_periods / total_periods * 100) if total_periods > 0 else 0
                        
                        sharpe_stats.append({
                            'Asset': asset,
                            'Current Sharpe': f"{data.iloc[-1]:.2f}",
                            'Avg Sharpe': f"{data.mean():.2f}",
                            'Min Sharpe': f"{data.min():.2f}",
                            'Max Sharpe': f"{data.max():.2f}",
                            'Good Rate (>1)': f"{good_rate:.1f}%"
                        })
                
                if sharpe_stats:
                    sharpe_df = pd.DataFrame(sharpe_stats)
                    st.dataframe(sharpe_df, use_container_width=True, hide_index=True)
            else:
                st.warning(f"Not enough data for {ROLLING_WINDOW_DAYS}-day rolling Sharpe ratio calculation.")
    except Exception as e:
        st.error(f"Error creating rolling Sharpe ratio chart: {e}")

with tab4:
    # Correlation Heatmap
    try:
        if 'correlation' in summary and not summary['correlation'].empty:
            corr_matrix = summary['correlation']

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.2f}<extra></extra>'
            ))

            fig_corr.update_layout(
                title="Correlation Matrix",
                height=400
            )

            st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating correlation chart: {e}")

with tab5:
    # Drawdown Analysis
    st.markdown("**📉 Drawdown Analysis**: Shows the decline from peak value. Maximum drawdown points are highlighted.")
    
    try:
        drawdown_data = {}
        max_dd_points = {}  # Store max drawdown points for annotations

        for asset in selected_assets:
            if asset in price_data.columns:
                prices = price_data[asset].dropna()
                peak = prices.expanding(min_periods=1).max()
                drawdown = ((prices - peak) / peak * 100)
                drawdown_data[asset] = drawdown
                
                # Find maximum drawdown point
                max_dd_idx = drawdown.idxmin()
                max_dd_value = drawdown.min()
                max_dd_points[asset] = {'date': max_dd_idx, 'value': max_dd_value}

        if drawdown_data:
            dd_df = pd.DataFrame(drawdown_data)

            fig_dd = go.Figure()

            for i, asset in enumerate(dd_df.columns):
                data = dd_df[asset].dropna()
                if len(data) > 0:
                    # Add drawdown line
                    fig_dd.add_trace(go.Scatter(
                        x=data.index,
                        y=data.values,
                        mode='lines',
                        name=asset,
                        fill='tozeroy',
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                        fillcolor=f"rgba({int(CHART_COLORS[i % len(CHART_COLORS)][1:3], 16)}, {int(CHART_COLORS[i % len(CHART_COLORS)][3:5], 16)}, {int(CHART_COLORS[i % len(CHART_COLORS)][5:7], 16)}, 0.2)",
                        hovertemplate=f'<b>{asset}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Drawdown: %{y:.1f}%<br>' +
                                    '<extra></extra>'
                    ))
                    
                    # Add marker for maximum drawdown
                    if asset in max_dd_points:
                        max_dd = max_dd_points[asset]
                        fig_dd.add_trace(go.Scatter(
                            x=[max_dd['date']],
                            y=[max_dd['value']],
                            mode='markers+text',
                            name=f'{asset} Max DD',
                            marker=dict(
                                size=12,
                                color=CHART_COLORS[i % len(CHART_COLORS)],
                                symbol='x',
                                line=dict(width=2, color='white')
                            ),
                            text=[f"{max_dd['value']:.1f}%"],
                            textposition='bottom center',
                            textfont=dict(size=10, color=CHART_COLORS[i % len(CHART_COLORS)]),
                            hovertemplate=f'<b>{asset} - Maximum Drawdown</b><br>' +
                                        f"Date: {max_dd['date'].strftime('%Y-%m-%d')}<br>" +
                                        f"Max DD: {max_dd['value']:.1f}%<br>" +
                                        '<extra></extra>',
                            showlegend=False
                        ))

            fig_dd.update_layout(
                title="Drawdown Analysis (Maximum Drawdown Points Highlighted)",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode="x unified",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            fig_dd.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            st.plotly_chart(fig_dd, use_container_width=True)
            
            # Add max drawdown summary table
            st.subheader("Maximum Drawdown Summary")
            max_dd_summary = []
            for asset, dd_info in max_dd_points.items():
                max_dd_summary.append({
                    'Asset': asset,
                    'Max Drawdown': f"{dd_info['value']:.2f}%",
                    'Date': dd_info['date'].strftime('%Y-%m-%d'),
                    'Days from Peak': (dd_info['date'] - price_data[asset].idxmax()).days
                })
            
            if max_dd_summary:
                max_dd_df = pd.DataFrame(max_dd_summary)
                st.dataframe(max_dd_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error creating drawdown chart: {e}")

# Save current settings to localStorage
save_settings_to_localStorage()
