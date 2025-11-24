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
    page_title="Asset Detail Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define colors
CHART_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Initialize processors
@st.cache_resource
def init_processors():
    return DataProcessor(), MetricsCalculator()

processor, calc = init_processors()

# Title
st.title("📊 Asset Detail Analysis")

# Load available assets
try:
    assets_df = processor.get_asset_list()
    all_assets = assets_df['asset_code'].tolist()
except Exception as e:
    st.error("Database not found. Please run data_processor.py first to load data.")
    st.stop()

# Sidebar for asset selection
st.sidebar.header("Asset Selection")

# Use session state if available, otherwise default to first asset
default_asset = st.session_state.get('selected_detail_asset', all_assets[0])
if default_asset not in all_assets:
    default_asset = all_assets[0]

default_index = all_assets.index(default_asset) if default_asset in all_assets else 0

selected_asset = st.sidebar.selectbox(
    "Choose Asset for Detailed Analysis:",
    options=all_assets,
    index=default_index,
    help="Select an asset to view detailed information and analysis"
)

if not selected_asset:
    st.warning("Please select an asset to analyze.")
    st.stop()

# Get asset details
asset_details = processor.get_asset_details(selected_asset)
if not asset_details:
    st.error(f"Asset details not found for {selected_asset}")
    st.stop()

# Display asset header
col1, col2 = st.columns([3, 1])
with col1:
    st.header(f"{selected_asset} - {asset_details.get('asset_name', 'N/A')}")
    asset_type_badge = "🏛️ Benchmark" if asset_details['asset_type'] == 'benchmark' else "💰 Fund"
    st.markdown(f"**{asset_type_badge}** | Inception: {asset_details.get('inception_date', 'N/A')}")

with col2:
    if st.button("🏠 Back to Dashboard", help="Return to main dashboard"):
        st.switch_page("dashboard.py")

# Asset Information Section
st.subheader("📋 Asset Information")

# Basic Info
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Description:**")
    st.write(asset_details.get('description', 'N/A'))

    st.markdown("**Investment Strategy:**")
    st.write(asset_details.get('investment_strategy', 'N/A'))

with col2:
    st.markdown("**Fund Size:**")
    st.write(asset_details.get('fund_size', 'N/A'))

    st.markdown("**Top Holdings:**")
    st.write(asset_details.get('top_holdings', 'N/A'))

# Sector Allocation
if asset_details.get('sector_allocation', 'N/A') != 'N/A':
    st.markdown("**Sector Allocation:**")
    st.write(asset_details.get('sector_allocation'))

st.divider()

# Price Analysis Section
st.subheader("💹 Price Analysis & Market Comparison")

# Date range selector
st.sidebar.subheader("Analysis Period")
col1, col2 = st.sidebar.columns(2)

# Get price data for date range
try:
    price_data_raw = processor.get_price_data([selected_asset])
    if price_data_raw.empty:
        st.warning(f"No price data available for {selected_asset}")
        st.stop()

    # Filter for selected asset and create proper time series
    asset_data = price_data_raw[price_data_raw['asset_code'] == selected_asset].copy()
    asset_data.set_index('date', inplace=True)
    price_data = asset_data['price']

    min_date = price_data.index.min().date()
    max_date = price_data.index.max().date()

except Exception as e:
    st.error(f"Error loading price data: {e}")
    st.stop()

with col1:
    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

# Filter data by selected date range
mask = (price_data.index.date >= start_date) & (price_data.index.date <= end_date)
filtered_price_data = price_data[mask]

if filtered_price_data.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

# Load VNIndex data for comparison (if not already selected asset)
vnindex_comparison = selected_asset != 'VNINDEX'
if vnindex_comparison:
    try:
        vnindex_data_raw = processor.get_price_data(['VNINDEX'], start_date, end_date)
        if not vnindex_data_raw.empty:
            vnindex_data_filtered = vnindex_data_raw[vnindex_data_raw['asset_code'] == 'VNINDEX'].copy()
            vnindex_data_filtered.set_index('date', inplace=True)
            vnindex_data = vnindex_data_filtered['price']
        else:
            vnindex_comparison = False
            vnindex_data = pd.Series(dtype=float)
    except:
        vnindex_comparison = False
        vnindex_data = pd.Series(dtype=float)

# Price Performance Charts
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Performance", "📊 VNIndex Comparison", "📉 Risk Metrics", "📋 Statistics"])

with tab1:
    # Price chart
    fig = go.Figure()

    asset_prices = filtered_price_data.dropna()
    fig.add_trace(go.Scatter(
        x=asset_prices.index,
        y=asset_prices.values,
        mode='lines',
        name=selected_asset,
        line=dict(color=CHART_COLORS[0], width=3),
        hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"{selected_asset} Price Performance",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Cumulative returns
    cumulative_returns = (asset_prices / asset_prices.iloc[0] * 100)

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns.values,
        mode='lines',
        name=f"{selected_asset} (Normalized)",
        line=dict(color=CHART_COLORS[1], width=3),
        hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Value: %{y:.1f}%<extra></extra>'
    ))

    fig_cum.update_layout(
        title=f"{selected_asset} Cumulative Returns (Base: 100%)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
        height=400
    )

    fig_cum.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig_cum, use_container_width=True)

with tab2:
    if vnindex_comparison and not vnindex_data.empty:
        # VNIndex comparison charts
        st.markdown("### Comparison with VN-Index")

        # Combined price chart (normalized)
        fig_comp = go.Figure()

        # Normalize both series to start at 100
        asset_norm = (asset_prices / asset_prices.iloc[0] * 100)
        vnindex_norm = (vnindex_data / vnindex_data.iloc[0] * 100)

        fig_comp.add_trace(go.Scatter(
            x=asset_norm.index,
            y=asset_norm.values,
            mode='lines',
            name=selected_asset,
            line=dict(color=CHART_COLORS[0], width=3)
        ))

        fig_comp.add_trace(go.Scatter(
            x=vnindex_norm.index,
            y=vnindex_norm.values,
            mode='lines',
            name='VN-Index',
            line=dict(color=CHART_COLORS[1], width=3)
        ))

        fig_comp.update_layout(
            title=f"{selected_asset} vs VN-Index (Normalized Performance)",
            xaxis_title="Date",
            yaxis_title="Normalized Value (%)",
            hovermode="x unified",
            height=400
        )

        fig_comp.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_comp, use_container_width=True)

        # Calculate correlation and beta
        try:
            # Align the data and calculate returns
            combined_data = pd.concat([asset_prices, vnindex_data], axis=1).dropna()

            if len(combined_data) > 1:
                returns_data = combined_data.pct_change().dropna()

                correlation = returns_data.iloc[:, 0].corr(returns_data.iloc[:, 1])

                # Calculate beta
                covariance = returns_data.iloc[:, 0].cov(returns_data.iloc[:, 1])
                market_variance = returns_data.iloc[:, 1].var()
                beta = covariance / market_variance if market_variance != 0 else 0

                # Display metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Correlation with VN-Index", f"{correlation:.3f}")

                with col2:
                    st.metric("Beta (vs VN-Index)", f"{beta:.3f}")

                with col3:
                    # Calculate relative performance
                    total_asset_return = ((asset_prices.iloc[-1] - asset_prices.iloc[0]) / asset_prices.iloc[0]) * 100
                    total_market_return = ((vnindex_data.iloc[-1] - vnindex_data.iloc[0]) / vnindex_data.iloc[0]) * 100
                    relative_performance = total_asset_return - total_market_return

                    st.metric("Relative Performance", f"{relative_performance:.2f}%")

                # Scatter plot for beta visualization
                st.markdown("### Beta Analysis")

                fig_beta = go.Figure()
                fig_beta.add_trace(go.Scatter(
                    x=returns_data.iloc[:, 1] * 100,
                    y=returns_data.iloc[:, 0] * 100,
                    mode='markers',
                    name='Daily Returns',
                    marker=dict(color=CHART_COLORS[0], opacity=0.6),
                    hovertemplate='VN-Index Return: %{x:.2f}%<br>' + f'{selected_asset} Return: %{y:.2f}%<extra></extra>'
                ))

                # Add trend line
                if not np.isnan(beta) and np.isfinite(beta):
                    x_min, x_max = returns_data.iloc[:, 1].min() * 100, returns_data.iloc[:, 1].max() * 100
                    x_trend = np.array([x_min, x_max])
                    y_trend = beta * x_trend
                    fig_beta.add_trace(go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        name=f'Beta = {beta:.3f}',
                        line=dict(color='red', width=2, dash='dash')
                    ))

                fig_beta.update_layout(
                    title="Beta Analysis - Daily Returns Correlation",
                    xaxis_title="VN-Index Daily Return (%)",
                    yaxis_title=f"{selected_asset} Daily Return (%)",
                    height=400
                )

                st.plotly_chart(fig_beta, use_container_width=True)

        except Exception as e:
            st.error(f"Error calculating comparison metrics: {e}")
    else:
        st.info("VN-Index comparison not available for this asset or date range.")

with tab3:
    # Risk metrics
    st.markdown("### Risk Analysis")

    try:
        # Calculate risk metrics
        returns = asset_prices.pct_change().dropna()

        if len(returns) > 0:
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0

            # Drawdown calculation
            peak = asset_prices.expanding(min_periods=1).max()
            drawdown = (asset_prices - peak) / peak * 100
            max_drawdown = drawdown.min()

            # Display metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Annualized Volatility", f"{volatility:.2f}%")

            with col2:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")

            with col3:
                st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")

            # Drawdown chart
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color=CHART_COLORS[2]),
                fillcolor=f'rgba(44, 160, 44, 0.2)',
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ))

            fig_dd.update_layout(
                title="Drawdown Analysis",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=400
            )

            fig_dd.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_dd, use_container_width=True)

    except Exception as e:
        st.error(f"Error calculating risk metrics: {e}")

with tab4:
    # Detailed statistics
    st.markdown("### Price Statistics")

    try:
        # Price statistics
        stats_data = {
            'Metric': [
                'Start Price',
                'End Price',
                'Minimum Price',
                'Maximum Price',
                'Average Price',
                'Total Return',
                'Annualized Return'
            ],
            'Value': [
                f"{asset_prices.iloc[0]:.2f}",
                f"{asset_prices.iloc[-1]:.2f}",
                f"{asset_prices.min():.2f}",
                f"{asset_prices.max():.2f}",
                f"{asset_prices.mean():.2f}",
                f"{((asset_prices.iloc[-1] - asset_prices.iloc[0]) / asset_prices.iloc[0] * 100):.2f}%",
                f"{(((asset_prices.iloc[-1] / asset_prices.iloc[0]) ** (365.25 / len(asset_prices))) - 1) * 100:.2f}%" if len(asset_prices) > 1 else "N/A"
            ]
        }

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Data availability
        st.markdown("### Data Availability")
        data_info = {
            'Metric': [
                'Data Points',
                'First Date',
                'Last Date',
                'Date Range (Days)',
                'Data Completeness'
            ],
            'Value': [
                len(asset_prices),
                asset_prices.index[0].strftime('%d/%m/%Y'),
                asset_prices.index[-1].strftime('%d/%m/%Y'),
                (asset_prices.index[-1] - asset_prices.index[0]).days,
                f"{(len(asset_prices) / ((asset_prices.index[-1] - asset_prices.index[0]).days + 1) * 100):.1f}%"
            ]
        }

        data_df = pd.DataFrame(data_info)
        st.dataframe(data_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating statistics: {e}")

# Footer
st.markdown("---")
st.markdown("**Asset Detail Analysis** | Navigate back to [Dashboard](/) for portfolio comparison")