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
            metrics_df[['asset_code', 'return_1y', 'return_3y_annualized', 'sharpe_ratio', 'max_drawdown_3y', 'calculated_at']],
            on='asset_code',
            how='left'
        )
    else:
        # Add empty metric columns
        assets_df['return_1y'] = None
        assets_df['return_3y_annualized'] = None
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
    
    # Function to display asset section
    def display_asset_section(title, emoji, df, section_color):
        if len(df) > 0:
            st.subheader(f"{emoji} {title}")
            st.markdown(f"*{len(df)} asset(s) available*")
            
            # Create cards for each asset
            cols_per_row = 3
            rows = (len(df) + cols_per_row - 1) // cols_per_row
            
            for row in range(rows):
                cols = st.columns(cols_per_row)
                
                for col_idx in range(cols_per_row):
                    asset_idx = row * cols_per_row + col_idx
                    
                    if asset_idx < len(df):
                        asset = df.iloc[asset_idx]
                        
                        with cols[col_idx]:
                            # Create a card-like container
                            with st.container():
                                # Format metrics
                                metrics_html = ""
                                if pd.notna(asset.get('return_1y')):
                                    return_1y = asset['return_1y'] * 100
                                    return_color = "#28a745" if return_1y >= 0 else "#dc3545"
                                    metrics_html += f'<p style="margin: 5px 0;"><strong>1Y Return:</strong> <span style="color: {return_color};">{return_1y:.2f}%</span></p>'
                                
                                if pd.notna(asset.get('return_3y_annualized')):
                                    return_3y = asset['return_3y_annualized'] * 100
                                    return_color = "#28a745" if return_3y >= 0 else "#dc3545"
                                    metrics_html += f'<p style="margin: 5px 0;"><strong>3Y Return:</strong> <span style="color: {return_color};">{return_3y:.2f}%</span></p>'
                                
                                if pd.notna(asset.get('sharpe_ratio')):
                                    sharpe = asset['sharpe_ratio']
                                    sharpe_color = "#28a745" if sharpe >= 0.5 else "#ffc107" if sharpe >= 0 else "#dc3545"
                                    metrics_html += f'<p style="margin: 5px 0;"><strong>Sharpe:</strong> <span style="color: {sharpe_color};">{sharpe:.2f}</span></p>'
                                
                                if pd.notna(asset.get('max_drawdown_3y')):
                                    max_dd = asset['max_drawdown_3y'] * 100
                                    metrics_html += f'<p style="margin: 5px 0;"><strong>Max DD (3Y):</strong> <span style="color: #dc3545;">{max_dd:.2f}%</span></p>'
                                
                                if not metrics_html:
                                    metrics_html = '<p style="margin: 5px 0; color: #999; font-style: italic;">No metrics calculated</p>'
                                
                                st.markdown(f"""
                                <div style="
                                    padding: 20px;
                                    border-radius: 10px;
                                    background: linear-gradient(135deg, {section_color}15 0%, {section_color}05 100%);
                                    border-left: 4px solid {section_color};
                                    margin-bottom: 10px;
                                    min-height: 220px;
                                ">
                                    <h3 style="margin-top: 0; color: {section_color};">{asset['asset_code']}</h3>
                                    <p style="margin: 5px 0;"><strong>Type:</strong> {asset['asset_type'].title()}</p>
                                    <p style="margin: 5px 0;"><strong>Inception:</strong> {asset['inception_date'].strftime('%Y-%m-%d')}</p>
                                    <p style="margin: 5px 0;"><strong>Records:</strong> {asset['record_count']:,}</p>
                                    <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
                                    {metrics_html}
                                    <p style="margin: 5px 0; color: #666; font-size: 0.85em;">
                                        Since {asset['inception_date'].year} • {(datetime.now() - asset['inception_date']).days // 365} years
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Add button to view details
                                if st.button(
                                    f"📊 View Details",
                                    key=f"detail_{asset['asset_code']}",
                                    use_container_width=True
                                ):
                                    st.session_state['selected_detail_asset'] = asset['asset_code']
                                    st.switch_page("pages/2_📊_Asset_Detail.py")
            
            st.markdown("---")
    
    # Display each section
    display_asset_section("Mutual Funds", "🏦", funds_df, "#1f77b4")
    display_asset_section("ETFs", "📈", etfs_df, "#ff7f0e")
    display_asset_section("Benchmarks", "📊", benchmarks_df, "#2ca02c")
    
    # Add a quick comparison section
    st.subheader("🔍 Quick Comparison")
    st.markdown("Select multiple assets to compare on the main dashboard")
    
    # Create a simple table for quick reference with metrics
    columns_to_include = ['asset_code', 'asset_type', 'inception_date', 'record_count', 
                          'return_1y', 'return_3y_annualized', 'return_5y_annualized', 
                          'sharpe_ratio', 'max_drawdown_3y']
    comparison_df = assets_df[columns_to_include].copy()
    
    # Format dates and percentages
    comparison_df['inception_date'] = comparison_df['inception_date'].dt.strftime('%Y-%m-%d')
    
    # Format percentage columns
    for col in ['return_1y', 'return_3y_annualized', 'return_5y_annualized', 'max_drawdown_3y']:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )
    
    # Format Sharpe ratio
    if 'sharpe_ratio' in comparison_df.columns:
        comparison_df['sharpe_ratio'] = comparison_df['sharpe_ratio'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
    
    # Rename columns for display
    comparison_df.columns = ['Asset Code', 'Type', 'Inception', 'Records', 
                             '1Y Return', '3Y Return (p.a.)', '5Y Return (p.a.)', 
                             'Sharpe Ratio', 'Max DD (3Y)']
    
    # Convert to proper types to avoid Arrow serialization issues
    for col in comparison_df.columns:
        comparison_df[col] = comparison_df[col].astype(str)
    
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
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
