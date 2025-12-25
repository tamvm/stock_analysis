import streamlit as st
import pandas as pd
import json
import time
from data_processor import DataProcessor
from datetime import datetime
from metrics_job_manager import MetricsJobManager

# Page config
st.set_page_config(
    page_title="Assets List - Investment Analysis Portal",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize processor and job manager
@st.cache_resource
def init_processor():
    return DataProcessor()

@st.cache_resource
def init_job_manager():
    return MetricsJobManager()

processor = init_processor()
job_manager = init_job_manager()

# Initialize session state for calculation progress
if 'calculation_running' not in st.session_state:
    st.session_state.calculation_running = False
if 'calculation_progress' not in st.session_state:
    st.session_state.calculation_progress = {'current': 0, 'total': 0, 'asset': None, 'status': 'idle'}

# Title
st.title("📋 Assets List")
st.markdown("Browse all available assets organized by type")

# Metrics Calculation Section
st.markdown("### 📊 Asset Metrics")

col1, col2 = st.columns([3, 1])

with col1:
    # Show last calculation time
    last_run = job_manager.get_last_calculation_run()
    if last_run:
        last_calc_time = pd.to_datetime(last_run['completed_at']).strftime('%Y-%m-%d %H:%M:%S')
        st.info(f"📅 Last calculation: {last_calc_time} ({last_run['assets_processed']}/{last_run['assets_total']} assets)")
    else:
        st.info("📅 No metrics calculated yet. Click the button to calculate.")

with col2:
    # Calculate button
    if st.button(
        "🔄 Calculate Metrics",
        disabled=st.session_state.calculation_running,
        use_container_width=True,
        type="primary"
    ):
        st.session_state.calculation_running = True
        st.session_state.calculation_progress = {'current': 0, 'total': 0, 'asset': None, 'status': 'running'}
        
        # Define progress callback
        def progress_callback(asset_code, processed, total, status, error):
            st.session_state.calculation_progress = {
                'current': processed,
                'total': total,
                'asset': asset_code,
                'status': status,
                'error': error
            }
            
        # Start background job
        job_manager.start_calculation_job(progress_callback)
        st.rerun()

# Show progress if calculation is running
if st.session_state.calculation_running:
    progress = st.session_state.calculation_progress
    
    if progress['status'] == 'running':
        st.progress(progress['current'] / max(progress['total'], 1))
        if progress['asset']:
            st.text(f"Processing {progress['asset']}... ({progress['current']}/{progress['total']})")
    elif progress['status'] == 'completed':
        st.success(f"✅ Calculation completed! Processed {progress['current']}/{progress['total']} assets")
        if progress['error']:
            st.warning(f"⚠️ Some errors occurred: {progress['error']}")
        st.session_state.calculation_running = False
        time.sleep(2)  # Brief pause before clearing
        st.rerun()

st.markdown("---")


# Load asset list
try:
    assets_df = processor.get_asset_list()
    
    # Get price data to calculate record counts
    all_asset_codes = assets_df['asset_code'].tolist()
    price_data = processor.get_price_data(all_asset_codes)
    
    # Calculate record counts for each asset
    record_counts = price_data.groupby('asset_code').size().to_dict()
    
    # Add record count to assets_df
    assets_df['record_count'] = assets_df['asset_code'].map(record_counts).fillna(0).astype(int)
    
    # Convert inception_date to datetime for proper formatting
    assets_df['inception_date'] = pd.to_datetime(assets_df['inception_date'])
    
    # Load metrics data
    metrics_df = job_manager.get_all_metrics()
    if len(metrics_df) > 0:
        # Merge metrics with assets
        assets_df = assets_df.merge(
            metrics_df[['asset_code', 'return_1y', 'return_3y_annualized', 'return_5y_annualized', 
                        'sharpe_ratio', 'max_drawdown_3y', 'calculated_at']],
            on='asset_code',
            how='left'
        )
    else:
        # Add empty metric columns
        assets_df['return_1y'] = None
        assets_df['return_3y_annualized'] = None
        assets_df['return_5y_annualized'] = None
        assets_df['sharpe_ratio'] = None
        assets_df['max_drawdown_3y'] = None
        assets_df['calculated_at'] = None
    
    # Group assets by type
    funds_df = assets_df[assets_df['asset_type'] == 'fund'].copy()
    etfs_df = assets_df[assets_df['asset_type'] == 'etf'].copy()
    benchmarks_df = assets_df[assets_df['asset_type'] == 'benchmark'].copy()
    
    # Display summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assets", len(assets_df))
    with col2:
        st.metric("Mutual Funds", len(funds_df))
    with col3:
        st.metric("ETFs", len(etfs_df))
    with col4:
        st.metric("Benchmarks", len(benchmarks_df))
    
    st.markdown("---")
    
    # Asset Comparison Table with Filters
    st.subheader("🔍 Asset Comparison Table")
    st.markdown("Filter and sort assets to compare metrics side-by-side")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Asset type filter
        asset_types = ['All'] + sorted(assets_df['asset_type'].unique().tolist())
        selected_type = st.selectbox("Filter by Type", asset_types, key="type_filter")
    
    with col2:
        # Metrics availability filter
        metrics_filter = st.selectbox(
            "Metrics Status",
            ["All", "With Metrics", "Without Metrics"],
            key="metrics_filter"
        )
    
    with col3:
        # Search by asset code
        search_term = st.text_input("Search Asset Code", "", key="search_filter")
    
    # Apply filters
    filtered_df = assets_df.copy()
    
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['asset_type'] == selected_type]
    
    if metrics_filter == "With Metrics":
        filtered_df = filtered_df[filtered_df['return_1y'].notna()]
    elif metrics_filter == "Without Metrics":
        filtered_df = filtered_df[filtered_df['return_1y'].isna()]
    
    if search_term:
        filtered_df = filtered_df[filtered_df['asset_code'].str.contains(search_term, case=False, na=False)]
    
    # Prepare display dataframe
    display_df = filtered_df.copy()
    
    # Format columns for display
    display_df['inception_date'] = display_df['inception_date'].dt.strftime('%Y-%m-%d')
    
    # Format percentage columns
    for col in ['return_1y', 'return_3y_annualized', 'return_5y_annualized', 'max_drawdown_3y']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )
    
    # Format Sharpe ratio
    if 'sharpe_ratio' in display_df.columns:
        display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
    
    # Format calculated_at timestamp
    if 'calculated_at' in display_df.columns:
        display_df['calculated_at'] = pd.to_datetime(display_df['calculated_at'], errors='coerce')
        display_df['calculated_at'] = display_df['calculated_at'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else "N/A"
        )
    
    # Select and rename columns for display
    columns_to_display = {
        'asset_code': 'Asset Code',
        'asset_type': 'Type',
        'inception_date': 'Inception',
        'record_count': 'Records',
        'return_1y': '1Y Return',
        'return_3y_annualized': '3Y Return (p.a.)',
        'return_5y_annualized': '5Y Return (p.a.)',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown_3y': 'Max DD (3Y)',
        'calculated_at': 'Last Calculated'
    }
    
    display_df = display_df[list(columns_to_display.keys())].rename(columns=columns_to_display)
    
    # Convert all to string to avoid Arrow serialization issues
    for col in display_df.columns:
        display_df[col] = display_df[col].astype(str)
    
    # Display count
    st.info(f"📊 Showing {len(display_df)} of {len(assets_df)} assets")
    
    # Display the dataframe with enhanced features
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600  # Fixed height for better scrolling
    )
    
    # Export option
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("💡 **Tip:** Click on column headers to sort. Use filters above to narrow down assets.")
    
    with col2:
        # Download button for CSV export
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"assets_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("dashboard.py")
    
    with col2:
        if st.button("📚 Metrics Guide", use_container_width=True):
            st.switch_page("pages/1_📚_Metrics_Guide.py")
    
    with col3:
        if st.button("🔍 Advanced Filter", use_container_width=True):
            st.switch_page("pages/3_🔍_Advanced_Filter.py")

except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.info("Please ensure the database has been initialized by running `python setup.py`")

