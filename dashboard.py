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
# Version tag: increment this to force cache refresh when calculation logic changes
METRICS_VERSION = "v3.0_independent_asset_processing"

@st.cache_resource
def init_processors(_version=METRICS_VERSION):
    """Initialize data processor and metrics calculator
    
    Args:
        _version: Version tag to force cache invalidation (prefix with _ to exclude from hash)
    """
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
    # US funds
    us_fund_assets = assets_df[assets_df['asset_type'] == 'us_fund']['asset_code'].tolist()
    # US ETFs
    etf_assets = assets_df[assets_df['asset_type'] == 'us_etf']['asset_code'].tolist()
    # Benchmarks include: benchmark type, vn_index type
    benchmark_assets = assets_df[assets_df['asset_type'].isin(['benchmark', 'vn_index'])]['asset_code'].tolist()
    # Also get other asset types
    stock_assets = assets_df[assets_df['asset_type'] == 'us_stock']['asset_code'].tolist()
    crypto_assets = assets_df[assets_df['asset_type'] == 'crypto']['asset_code'].tolist()
    commodity_assets = assets_df[assets_df['asset_type'] == 'commodity']['asset_code'].tolist()
    
    all_assets = fund_assets + us_fund_assets + etf_assets + benchmark_assets + stock_assets + crypto_assets + commodity_assets
except:
    st.error("Database not found. Please run data_processor.py first to load data.")
    st.stop()

# Get URL query parameters for all filters
query_params = st.query_params
url_assets = query_params.get('assets', None)
url_rolling_years = query_params.get('rolling_years', None)
url_analysis_periods = query_params.get('periods', None)
url_start_date = query_params.get('start_date', None)
url_end_date = query_params.get('end_date', None)

# Parse URL assets if present (comma-separated)
url_selected_assets = []
if url_assets:
    url_selected_assets = [asset.strip() for asset in url_assets.split(',') if asset.strip() in all_assets]

# Parse URL rolling years
url_rolling_years_value = None
if url_rolling_years:
    try:
        url_rolling_years_value = int(url_rolling_years)
    except (ValueError, TypeError):
        url_rolling_years_value = None

# Parse URL analysis periods (comma-separated)
url_analysis_periods_list = []
if url_analysis_periods:
    url_analysis_periods_list = [period.strip() for period in url_analysis_periods.split(',') if period.strip() in ['1Y', '3Y', '5Y', 'Inception']]

# Parse URL dates (format: YYYY-MM-DD)
url_start_date_value = None
url_end_date_value = None
if url_start_date:
    try:
        url_start_date_value = datetime.strptime(url_start_date, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        url_start_date_value = None

if url_end_date:
    try:
        url_end_date_value = datetime.strptime(url_end_date, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        url_end_date_value = None

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
    'us_fund': us_fund_assets,
    'us_etf': etf_assets,
    'us_stock': stock_assets,
    'crypto': crypto_assets,
    'commodity': commodity_assets,
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
                    # Directly update the multiselect widget's session state value
                    st.session_state.selected_assets_multiselect = valid_preset_assets
                    # Also update our tracking of saved assets
                    st.session_state.saved_selected_assets = valid_preset_assets
                    # Update URL query parameters
                    if valid_preset_assets:
                        st.query_params['assets'] = ','.join(valid_preset_assets)
                    # Trigger a rerun to update the UI
                    st.rerun()

# Determine the default value for multiselect
# Priority: 1) Existing widget value (if already set), 2) URL parameters, 3) Preset selection, 4) Saved assets, 5) VN Top Funds preset
# Only set default if the widget hasn't been initialized yet
if 'selected_assets_multiselect' in st.session_state and st.session_state.selected_assets_multiselect is not None:
    # Widget already has a value, use it (this prevents reset on rerun)
    default_assets = st.session_state.selected_assets_multiselect
elif url_selected_assets:
    # Use assets from URL parameters
    default_assets = url_selected_assets
elif st.session_state.preset_selected_assets is not None:
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

# Callback function to handle asset selection changes
def on_assets_change():
    """Update session state and URL parameters when assets are selected"""
    selected = st.session_state.selected_assets_multiselect
    st.session_state.selected_assets_ls = selected
    st.session_state.saved_selected_assets = selected
    
    # Update URL query parameters
    if selected:
        st.query_params['assets'] = ','.join(selected)
    else:
        # Remove assets parameter if no assets selected
        if 'assets' in st.query_params:
            del st.query_params['assets']

# Callback function to handle rolling years changes
def on_rolling_years_change():
    """Update session state and URL parameters when rolling years is changed"""
    rolling_years = st.session_state.rolling_years_selectbox
    st.session_state.rolling_years_ls = rolling_years
    st.query_params['rolling_years'] = str(rolling_years)

# Callback function to handle analysis periods changes
def on_analysis_periods_change():
    """Update session state and URL parameters when analysis periods are changed"""
    periods = st.session_state.analysis_periods_multiselect
    st.session_state.analysis_periods_ls = periods
    
    if periods:
        st.query_params['periods'] = ','.join(periods)
    else:
        if 'periods' in st.query_params:
            del st.query_params['periods']

# Callback function to handle date changes
def on_date_change():
    """Update session state and URL parameters when dates are changed"""
    start = st.session_state.start_date
    end = st.session_state.end_date
    
    st.session_state.start_date_ls = start
    st.session_state.end_date_ls = end
    
    # Update URL parameters
    if start:
        st.query_params['start_date'] = start.strftime('%Y-%m-%d')
    else:
        if 'start_date' in st.query_params:
            del st.query_params['start_date']
    
    if end:
        st.query_params['end_date'] = end.strftime('%Y-%m-%d')
    else:
        if 'end_date' in st.query_params:
            del st.query_params['end_date']

# Asset selection
st.sidebar.subheader("Select Assets to Compare")
selected_assets = st.sidebar.multiselect(
    "Choose Assets:",
    options=all_assets,
    default=default_assets,
    help="Select multiple assets to compare. Includes funds, ETFs, and benchmarks.",
    key="selected_assets_multiselect",
    on_change=on_assets_change
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


# Handle URL parameters and localStorage settings for rolling years
# Priority: 1) URL parameter, 2) localStorage
saved_rolling_years = loaded_settings.get('rolling_years', 4)
rolling_years_options = [1, 2, 3, 4, 5]

# Determine default rolling years
default_rolling_years = saved_rolling_years
if url_rolling_years_value is not None and url_rolling_years_value in rolling_years_options:
    default_rolling_years = url_rolling_years_value

# Find the index of default value, default to 2 (4 years) if not found
try:
    rolling_years_index = rolling_years_options.index(default_rolling_years)
except ValueError:
    rolling_years_index = 2  # Default to 4 years

# Rolling period selector
st.sidebar.subheader("Rolling CAGR Settings")
rolling_years = st.sidebar.selectbox(
    "Rolling CAGR Period:",
    options=rolling_years_options,
    index=rolling_years_index,
    help="Select the rolling period for CAGR calculation",
    key="rolling_years_selectbox",
    on_change=on_rolling_years_change
)

# Update session state for localStorage saving
st.session_state.rolling_years_ls = rolling_years

# Handle URL parameters and localStorage settings for analysis periods
# Priority: 1) URL parameter, 2) localStorage
saved_analysis_periods = loaded_settings.get('analysis_periods', ['1Y', '3Y', '5Y'])
analysis_options = ['1Y', '3Y', '5Y', 'Inception']

# Determine default analysis periods
default_analysis_periods = saved_analysis_periods
if url_analysis_periods_list:
    default_analysis_periods = url_analysis_periods_list

# Filter default periods to only include valid ones
valid_default_periods = [period for period in default_analysis_periods if period in analysis_options]

# Performance period selector
st.sidebar.subheader("Performance Analysis")
analysis_periods = st.sidebar.multiselect(
    "Analysis Periods:",
    options=analysis_options,
    default=valid_default_periods if valid_default_periods else ['1Y', '3Y', '5Y'],
    help="Select time periods for performance comparison",
    key="analysis_periods_multiselect",
    on_change=on_analysis_periods_change
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
                
                # Update URL parameters
                st.query_params['start_date'] = earliest_date.strftime('%Y-%m-%d')
                st.query_params['end_date'] = latest_date.strftime('%Y-%m-%d')
                
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
                
                # Update URL parameters
                st.query_params['start_date'] = latest_inception_date.strftime('%Y-%m-%d')
                st.query_params['end_date'] = latest_date.strftime('%Y-%m-%d')
                
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error setting min inception range: {e}")

# Handle URL parameters and localStorage settings for date range
# Priority: 1) URL parameter, 2) localStorage
saved_start_date = loaded_settings.get('start_date', '')
saved_end_date = loaded_settings.get('end_date', '')

# Convert saved date strings back to date objects if they exist and are valid
start_date_value = None
end_date_value = None

# Use URL parameters if available, otherwise use localStorage
if url_start_date_value is not None:
    start_date_value = url_start_date_value
elif saved_start_date:
    try:
        start_date_value = datetime.strptime(saved_start_date, '%d/%m/%Y').date()
    except (ValueError, TypeError):
        start_date_value = None

if url_end_date_value is not None:
    end_date_value = url_end_date_value
elif saved_end_date:
    try:
        end_date_value = datetime.strptime(saved_end_date, '%d/%m/%Y').date()
    except (ValueError, TypeError):
        end_date_value = None

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=start_date_value,
        min_value=datetime(2000, 1, 1).date(),
        max_value=datetime.today().date(),
        key="start_date",
        on_change=on_date_change
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=end_date_value,
        min_value=datetime(2000, 1, 1).date(),
        max_value=datetime.today().date(),
        key="end_date",
        on_change=on_date_change
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
        # Calculate a smart ranking that only compares assets when they both have data
        # This gives a weighted average rank for each asset
        asset_weighted_rank = {}
        
        for asset in rolling_cagr.columns:
            asset_data = rolling_cagr[asset].dropna()
            
            if len(asset_data) > 0:
                total_weighted_rank = 0
                total_weight = 0
                
                # For each date where this asset has data
                for date in asset_data.index:
                    # Get all assets that have data at this date
                    values_at_date = rolling_cagr.loc[date].dropna()
                    
                    if len(values_at_date) >= 2:  # Need at least 2 assets to compare
                        # Calculate rank at this date (1 = highest CAGR, n = lowest)
                        asset_value = values_at_date[asset]
                        rank = (values_at_date > asset_value).sum() + 1
                        
                        # Weight by number of assets being compared (more assets = more meaningful)
                        weight = len(values_at_date)
                        total_weighted_rank += rank * weight
                        total_weight += weight
                
                # Calculate weighted average rank (lower is better)
                if total_weight > 0:
                    asset_weighted_rank[asset] = total_weighted_rank / total_weight
                else:
                    asset_weighted_rank[asset] = float('inf')
            else:
                asset_weighted_rank[asset] = float('inf')  # No data
        
        # Sort assets by weighted average rank (lower = better = appears first)
        sorted_assets = sorted(asset_weighted_rank.keys(), key=lambda x: asset_weighted_rank[x])
        
        # Create the chart with sorted traces
        fig_cagr = go.Figure()
        
        for asset in sorted_assets:
            # Find the original index for color consistency
            original_index = list(rolling_cagr.columns).index(asset)
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
                    line=dict(color=CHART_COLORS[original_index % len(CHART_COLORS)], width=3),
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

            # Use the same sorted order as the chart
            for asset in sorted_assets:
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
            
            # Use the same sorted order as the chart
            for asset in sorted_assets:
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
            
            # Use the same sorted order as the chart
            for asset in sorted_assets:
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

# CAGR Table Section
st.subheader("📊 Compound Annual Growth Rate (CAGR) Table")

try:
    # Determine the end date for CAGR calculation
    # Priority: 1) User-selected end_date, 2) Latest date in database
    cagr_end_date = end_date if end_date else None
    
    # Calculate CAGR table
    cagr_df = calc.calculate_cagr_table(selected_assets, end_date=cagr_end_date)
    
    if not cagr_df.empty:
        # Get the actual end date used
        price_data_for_date, _ = calc.get_returns_data(selected_assets)
        if cagr_end_date:
            price_data_for_date = price_data_for_date[price_data_for_date.index <= pd.to_datetime(cagr_end_date)]
        actual_end_date = price_data_for_date.index.max()
        
        # Display date information
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Date:** {actual_end_date.strftime('%b %d, %Y')}")
        with col2:
            st.markdown(f"**Assets:** {len(selected_assets)}")
        
        # Prepare data for display with formatting
        display_data = []
        
        for asset in cagr_df.index:
            row = {'Asset': asset}
            
            for col in cagr_df.columns:
                value = cagr_df.loc[asset, col]
                if pd.isna(value) or value is None:
                    row[col] = ''
                else:
                    row[col] = value
            
            display_data.append(row)
        
        display_df = pd.DataFrame(display_data)
        
        # Function to style the dataframe
        def style_cagr_table(df):
            """Apply styling to CAGR table with color coding and star icons for top 3 performers"""
            
            # Create a new DataFrame with string values for display
            styled_data = []
            
            for idx in df.index:
                row = {'Asset': df.loc[idx, 'Asset']}
                
                for col in df.columns:
                    if col == 'Asset':
                        continue
                    
                    val = df.loc[idx, col]
                    
                    if val == '' or pd.isna(val):
                        row[col] = ''
                    else:
                        # Determine ranking for this column
                        col_values = df[col].replace('', np.nan).dropna()
                        star_icon = ''
                        
                        if len(col_values) > 0:
                            # Sort values in descending order to get rankings
                            sorted_values = col_values.sort_values(ascending=False)
                            
                            # Get unique values to handle ties
                            unique_values = sorted_values.unique()
                            
                            # Determine rank based on value
                            if len(unique_values) >= 1 and val == unique_values[0]:
                                star_icon = '⭐'  # 1st place
                            elif len(unique_values) >= 2 and val == unique_values[1]:
                                star_icon = '★'   # 2nd place
                        
                        # Format number with fixed width (right-aligned)
                        formatted_number = f"{val:+7.1f}%"
                        
                        # Combine icon and number with icon on left, number right-aligned
                        if star_icon:
                            row[col] = f"{star_icon} {formatted_number}"
                        else:
                            # Add spacing to align with cells that have icons
                            row[col] = f"  {formatted_number}"
                
                styled_data.append(row)
            
            return pd.DataFrame(styled_data)
        
        # Apply styling
        styled_display_df = style_cagr_table(display_df)
        
        # Function to apply color coding
        def color_cagr_cells(val):
            """Apply color based on positive/negative values"""
            if val == '' or pd.isna(val):
                return ''
            
            # Extract numeric value from formatted string (remove star and %)
            try:
                numeric_str = val.replace('⭐', '').replace('★', '').replace('%', '').strip()
                numeric_val = float(numeric_str)
                
                if numeric_val > 0:
                    return 'background-color: #d4edda; color: #155724; text-align: right; padding: 8px 12px; font-size: 14px;'  # Green
                elif numeric_val < 0:
                    return 'background-color: #f8d7da; color: #721c24; text-align: right; padding: 8px 12px; font-size: 14px;'  # Red
                else:
                    return 'text-align: right; padding: 8px 12px; font-size: 14px;'
            except (ValueError, AttributeError):
                return 'text-align: right; padding: 8px 12px; font-size: 14px;'
        
        # Function to style Asset column
        def style_asset_column(val):
            """Style the Asset column"""
            return 'text-align: left; padding: 8px 12px; font-size: 14px; font-weight: 600;'
        
        # Apply color styling to all columns except Asset
        styled_table = styled_display_df.style.map(
            color_cagr_cells,
            subset=[col for col in styled_display_df.columns if col != 'Asset']
        ).map(
            style_asset_column,
            subset=['Asset']
        )
        
        # Display the styled table
        st.dataframe(
            styled_table,
            use_container_width=True,
            height=min(500, (len(display_df) + 1) * 45 + 50)  # Increased height for bigger cells
        )
        
        # Add explanation
        st.markdown("""
        **📖 How to read this table:**
        - Each cell shows the Compound Annual Growth Rate (CAGR) for the specified period
        - ⭐ indicates the **1st place** (best performing asset) for that time period
        - ★ indicates the **2nd place** performer for that time period
        - 🟢 Green cells indicate positive returns
        - 🔴 Red cells indicate negative returns
        - Empty cells mean insufficient data for that period
        - Click column headers to sort
        """)
        
    else:
        st.warning("No CAGR data available for selected assets and date range.")
        
except Exception as e:
    st.error(f"Error calculating CAGR table: {e}")
    import traceback
    st.error(traceback.format_exc())

# Additional charts row
st.subheader("📊 Additional Analysis")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Cumulative Returns", "Price Changes", "Risk-Return", "Sharpe Ratio", "Correlation", "Drawdown"])

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
    # Sharpe Ratio Comparison
    st.markdown("**📊 Sharpe Ratio Comparison**: Compare risk-adjusted returns across assets and time periods. Higher Sharpe ratio indicates better risk-adjusted performance.")
    
    try:
        if analysis_periods and summary:
            # Collect Sharpe ratios for all periods
            sharpe_data = []
            
            for period in analysis_periods:
                if period in summary and 'sharpe_ratio' in summary[period]:
                    for asset in selected_assets:
                        sharpe = summary[period]['sharpe_ratio'].get(asset)
                        if isinstance(sharpe, (int, float, np.floating)) and not np.isnan(sharpe):
                            sharpe_data.append({
                                'Asset': asset,
                                'Period': period,
                                'Sharpe Ratio': sharpe
                            })
            
            if sharpe_data:
                sharpe_df = pd.DataFrame(sharpe_data)
                
                # Create grouped bar chart
                fig_sharpe = px.bar(
                    sharpe_df,
                    x='Asset',
                    y='Sharpe Ratio',
                    color='Period',
                    barmode='group',
                    title="Sharpe Ratio Comparison by Period",
                    labels={'Sharpe Ratio': 'Sharpe Ratio', 'Asset': 'Asset'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                
                # Add reference line at Sharpe = 1 (good performance threshold)
                fig_sharpe.add_hline(
                    y=1, 
                    line_dash="dash", 
                    line_color="green", 
                    opacity=0.5,
                    annotation_text="Good (>1)",
                    annotation_position="right"
                )
                
                # Add reference line at Sharpe = 0
                fig_sharpe.add_hline(
                    y=0, 
                    line_dash="dash", 
                    line_color="gray", 
                    opacity=0.5
                )
                
                fig_sharpe.update_layout(
                    height=500,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_sharpe, use_container_width=True)
                
                # Summary table
                st.markdown("### Sharpe Ratio Summary")
                
                # Create pivot table for easy comparison
                summary_data = []
                for asset in selected_assets:
                    row = {'Asset': asset}
                    for period in analysis_periods:
                        if period in summary and 'sharpe_ratio' in summary[period]:
                            sharpe = summary[period]['sharpe_ratio'].get(asset)
                            if isinstance(sharpe, (int, float, np.floating)) and not np.isnan(sharpe):
                                row[f'{period} Sharpe'] = f"{sharpe:.2f}"
                            else:
                                row[f'{period} Sharpe'] = 'N/A'
                    
                    # Add average Sharpe across periods
                    sharpe_values = []
                    for period in analysis_periods:
                        if period in summary and 'sharpe_ratio' in summary[period]:
                            sharpe = summary[period]['sharpe_ratio'].get(asset)
                            if isinstance(sharpe, (int, float, np.floating)) and not np.isnan(sharpe):
                                sharpe_values.append(sharpe)
                    
                    if sharpe_values:
                        row['Avg Sharpe'] = f"{np.mean(sharpe_values):.2f}"
                    else:
                        row['Avg Sharpe'] = 'N/A'
                    
                    summary_data.append(row)
                
                if summary_data:
                    summary_table = pd.DataFrame(summary_data)
                    st.dataframe(summary_table, use_container_width=True, hide_index=True)
                    
                    # Interpretation guide
                    st.markdown("""
                    **Sharpe Ratio Interpretation:**
                    - **< 0**: Poor - Returns below risk-free rate
                    - **0 to 1**: Acceptable - Returns above risk-free rate but low risk-adjusted performance
                    - **1 to 2**: Good - Solid risk-adjusted returns
                    - **2 to 3**: Very Good - Excellent risk-adjusted returns
                    - **> 3**: Exceptional - Outstanding risk-adjusted performance
                    """)
            else:
                st.warning("No Sharpe ratio data available for the selected periods.")
        else:
            st.warning("Please select analysis periods to view Sharpe ratio comparison.")
    except Exception as e:
        st.error(f"Error creating Sharpe ratio comparison: {e}")

with tab5:
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

with tab6:
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
