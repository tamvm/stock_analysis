import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from data_processor import DataProcessor
from metrics_calculator import MetricsCalculator

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

# Title
st.title("📈 Investment Analysis Portal")
st.markdown("Professional investment analysis and comparison tool")

# Sidebar for controls
st.sidebar.header("Asset Selection & Settings")

# Load available assets
try:
    assets_df = processor.get_asset_list()
    fund_assets = assets_df[assets_df['asset_type'] == 'fund']['asset_code'].tolist()
    benchmark_assets = assets_df[assets_df['asset_type'] == 'benchmark']['asset_code'].tolist()
    all_assets = fund_assets + benchmark_assets
except:
    st.error("Database not found. Please run data_processor.py first to load data.")
    st.stop()

# Asset selection
st.sidebar.subheader("Select Assets to Compare")
selected_assets = st.sidebar.multiselect(
    "Choose funds and benchmarks:",
    options=all_assets,
    default=fund_assets[:3] if len(fund_assets) >= 3 else fund_assets,
    help="Select multiple assets to compare. You can toggle assets on/off."
)

if not selected_assets:
    st.warning("Please select at least one asset to analyze.")
    st.stop()

# Rolling period selector
st.sidebar.subheader("Rolling CAGR Settings")
rolling_years = st.sidebar.selectbox(
    "Rolling CAGR Period:",
    options=[1, 3, 4, 5],
    index=2,  # Default to 4 years
    help="Select the rolling period for CAGR calculation"
)

# Performance period selector
st.sidebar.subheader("Performance Analysis")
analysis_periods = st.sidebar.multiselect(
    "Analysis Periods:",
    options=['1Y', '3Y', '5Y', 'Inception'],
    default=['3Y', '5Y'],
    help="Select time periods for performance comparison"
)

# Date range selector
st.sidebar.subheader("Date Range (Optional)")

# Add button to set to maximum inception
if st.sidebar.button("📅 Use Maximum Inception Range", help="Set date range from earliest fund inception to latest data"):
    try:
        # Get inception dates for selected assets
        assets_df = processor.get_asset_list()
        selected_assets_df = assets_df[assets_df['asset_code'].isin(selected_assets)]

        if not selected_assets_df.empty:
            earliest_inception = pd.to_datetime(selected_assets_df['inception_date']).min().date()
            latest_date = pd.to_datetime(selected_assets_df['last_update']).max().date()

            st.session_state.start_date_input = earliest_inception
            st.session_state.end_date_input = latest_date
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error setting inception range: {e}")

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=st.session_state.get('start_date_input', None))
with col2:
    end_date = st.date_input("End Date", value=st.session_state.get('end_date_input', None))

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
                vol = summary[period]['volatility'].get(asset, 'N/A')

                if isinstance(ann_ret, (int, float)):
                    row[f'{period} Return'] = f"{ann_ret:.1f}%"
                else:
                    row[f'{period} Return'] = 'N/A'

                if isinstance(vol, (int, float)):
                    row[f'{period} Vol'] = f"{vol:.1f}%"
                else:
                    row[f'{period} Vol'] = 'N/A'

        # Add overall metrics
        max_dd = summary['max_drawdown'].get(asset, 'N/A')
        sharpe = summary['sharpe_ratio'].get(asset, 'N/A')

        if isinstance(max_dd, (int, float)):
            row['Max DD'] = f"{max_dd:.1f}%"
        else:
            row['Max DD'] = 'N/A'

        if isinstance(sharpe, (int, float)):
            row['Sharpe'] = f"{sharpe:.2f}"
        else:
            row['Sharpe'] = 'N/A'

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

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
                        f"From: {start_date.strftime('%Y-%m-%d')} To: {date.strftime('%Y-%m-%d')}<br>" +
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

# Additional charts row
st.subheader("📊 Additional Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Cumulative Returns", "Risk-Return", "Correlation", "Drawdown"])

with tab1:
    # Cumulative Returns Chart
    try:
        # Normalize to 100% start
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
                                'Value: %{y:.1f}<br>' +
                                '<extra></extra>'
                ))

        fig_cum.update_layout(
            title="Cumulative Returns (Normalized to 100%)",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode="x unified",
            height=400
        )

        fig_cum.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)

        st.plotly_chart(fig_cum, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating cumulative returns chart: {e}")

with tab2:
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

with tab3:
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

with tab4:
    # Drawdown Analysis
    try:
        drawdown_data = {}

        for asset in selected_assets:
            if asset in price_data.columns:
                prices = price_data[asset].dropna()
                peak = prices.expanding(min_periods=1).max()
                drawdown = ((prices - peak) / peak * 100)
                drawdown_data[asset] = drawdown

        if drawdown_data:
            dd_df = pd.DataFrame(drawdown_data)

            fig_dd = go.Figure()

            for i, asset in enumerate(dd_df.columns):
                data = dd_df[asset].dropna()
                if len(data) > 0:
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

            fig_dd.update_layout(
                title="Drawdown Analysis",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode="x unified",
                height=400
            )

            fig_dd.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            st.plotly_chart(fig_dd, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating drawdown chart: {e}")

# Footer
st.markdown("---")
st.markdown("**Investment Analysis Portal** | Data updated weekly | For educational purposes only")