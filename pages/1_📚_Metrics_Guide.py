import streamlit as st

st.set_page_config(
    page_title="Metrics Guide",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Investment Metrics Guide")
st.markdown("Understanding the financial metrics used in investment analysis")

# Create tabs for different metric categories
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "⚠️ Risk", "🔄 Correlation", "📊 Technical"])

with tab1:
    st.header("📈 Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Return")
        st.markdown("""
        **Formula:** (Final Price - Initial Price) / Initial Price × 100%

        **What it means:** The absolute percentage gain or loss over a period.

        **Example:** If you invested \\$10,000 and it's now worth \\$12,000, your total return is 20%.

        **Use case:** Compare absolute performance between different time periods.
        """)

        st.subheader("CAGR (Compound Annual Growth Rate)")
        st.markdown("""
        **Formula:** ((Final Value / Initial Value)^(1/Years)) - 1

        **What it means:** The annualized return that smooths out volatility.

        **Example:** A fund growing from \\$10,000 to \\$14,641 over 4 years has a CAGR of 10%.

        **Use case:** Compare funds with different time periods on an equal basis.
        """)

    with col2:
        st.subheader("Rolling CAGR")
        st.markdown("""
        **What it means:** CAGR calculated over a moving time window (e.g., 4 years).

        **Why it matters:** Shows how performance evolves over time and identifies consistent performers.

        **Example:** 4-year rolling CAGR shows what your return would be if you held for any 4-year period.

        **Use case:** Identify entry/exit timing and performance consistency.
        """)

        st.subheader("Excess Return")
        st.markdown("""
        **Formula:** Fund Return - Benchmark Return

        **What it means:** How much better (or worse) a fund performs vs. the market.

        **Example:** If your fund returns 12% and VN-Index returns 8%, excess return is 4%.

        **Use case:** Evaluate if active management adds value over passive indexing.
        """)

with tab2:
    st.header("⚠️ Risk Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Volatility (Standard Deviation)")
        st.markdown("""
        **Formula:** Standard deviation of daily returns × √252

        **What it means:** How much returns vary from the average (annualized).

        **Example:** 15% volatility means returns typically vary ±15% from the average.

        **Interpretation:**
        - Low volatility (< 10%): Conservative, stable
        - Medium volatility (10-20%): Moderate risk
        - High volatility (> 20%): Aggressive, risky
        """)

        st.subheader("Maximum Drawdown")
        st.markdown("""
        **Formula:** (Trough Value - Peak Value) / Peak Value

        **What it means:** The largest peak-to-trough decline during any period.

        **Example:** If a fund drops from \\$12,000 to \\$9,000, max drawdown is -25%.

        **Use case:** Understand worst-case scenarios and your risk tolerance.
        """)

    with col2:
        st.subheader("Sharpe Ratio")
        st.markdown("""
        **Formula:** (Return - Risk-Free Rate) / Volatility

        **What it means:** Risk-adjusted return. Higher is better.

        **Interpretation:**
        - Sharpe < 1: Poor risk-adjusted returns
        - Sharpe 1-2: Good risk-adjusted returns
        - Sharpe > 2: Excellent risk-adjusted returns

        **Use case:** Compare funds with different risk levels on equal footing.
        """)

        st.subheader("Sortino Ratio")
        st.markdown("""
        **Formula:** (Return - Risk-Free Rate) / Downside Deviation

        **What it means:** Like Sharpe ratio but only penalizes downside volatility.

        **Why it matters:** Upside volatility is good, downside is bad. Sortino focuses on bad volatility.

        **Use case:** Better measure for asymmetric return distributions.
        """)

with tab3:
    st.header("🔄 Correlation & Diversification")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Coefficient")
        st.markdown("""
        **Range:** -1 to +1

        **What it means:**
        - +1: Perfect positive correlation (move together)
        - 0: No correlation (independent movement)
        - -1: Perfect negative correlation (move opposite)

        **Example:** Two funds with 0.8 correlation move in the same direction 80% of the time.

        **Use case:** Build diversified portfolios by combining low-correlation assets.
        """)

        st.subheader("Rolling Correlation")
        st.markdown("""
        **What it means:** Correlation calculated over moving time windows.

        **Why it matters:** Correlations change over time, especially during crises.

        **Example:** Funds may be uncorrelated during normal times but highly correlated during market crashes.

        **Use case:** Monitor if your diversification strategy remains effective.
        """)

    with col2:
        st.subheader("Diversification Benefits")
        st.markdown("""
        **Portfolio Risk Formula:**
        ```
        Portfolio Risk = √(w1²σ1² + w2²σ2² + 2w1w2σ1σ2ρ12)
        ```
        Where ρ12 is correlation between assets 1 and 2.

        **Key Insight:** Lower correlation → Lower portfolio risk for same expected return.

        **Practical Application:**
        - Combine domestic and international funds
        - Mix growth and value styles
        - Add bonds to equity portfolios
        """)

        st.subheader("Beta (Market Sensitivity)")
        st.markdown("""
        **Formula:** Covariance(Fund, Market) / Variance(Market)

        **What it means:**
        - Beta = 1: Moves with market
        - Beta > 1: More volatile than market
        - Beta < 1: Less volatile than market

        **Example:** Beta of 1.2 means 20% more volatile than the market.
        """)

with tab4:
    st.header("📊 Technical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cumulative Returns")
        st.markdown("""
        **Formula:** (Current Price / Starting Price - 1) × 100%

        **What it shows:** Total growth from a common starting point (normalized to 100%).

        **Use case:** Visual comparison of different funds' performance over time.

        **Example:** Starting at 100%, one fund at 150% means 50% total gain.
        """)

        st.subheader("Drawdown Recovery")
        st.markdown("""
        **What it measures:** Time taken to recover from peak-to-trough declines.

        **Formula:** Days from trough to new high

        **Why it matters:** Shows resilience and ability to bounce back from losses.

        **Use case:** Evaluate fund management quality during difficult periods.
        """)

    with col2:
        st.subheader("Risk-Return Scatter")
        st.markdown("""
        **X-axis:** Volatility (risk)
        **Y-axis:** Return

        **Efficient Frontier:** Upper-left region where you get maximum return for given risk.

        **What to look for:**
        - Funds in upper-left quadrant (high return, low risk)
        - Avoid lower-right quadrant (low return, high risk)
        """)

        st.subheader("Win Rate")
        st.markdown("""
        **Formula:** Number of positive periods / Total periods

        **What it means:** Percentage of time periods with positive returns.

        **Example:** 65% win rate means positive returns 65% of the time.

        **Use case:** Understand consistency and frequency of gains vs. losses.
        """)

# Additional information section
st.markdown("---")
st.header("🎯 Practical Investment Guidelines")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Conservative Portfolio")
    st.markdown("""
    **Target Metrics:**
    - Volatility: < 10%
    - Max Drawdown: < 15%
    - Sharpe Ratio: > 0.8
    - Correlation with market: < 0.7

    **Suitable for:** Risk-averse investors, near retirement
    """)

with col2:
    st.subheader("Balanced Portfolio")
    st.markdown("""
    **Target Metrics:**
    - Volatility: 10-15%
    - Max Drawdown: 15-25%
    - Sharpe Ratio: > 1.0
    - Beta: 0.8-1.2

    **Suitable for:** Most long-term investors
    """)

with col3:
    st.subheader("Aggressive Portfolio")
    st.markdown("""
    **Target Metrics:**
    - Volatility: > 15%
    - Max Drawdown: > 25%
    - Sharpe Ratio: > 1.2
    - Beta: > 1.0

    **Suitable for:** Young investors, high risk tolerance
    """)

st.markdown("---")
st.info("💡 **Remember:** Past performance doesn't guarantee future results. Use these metrics as guides, not absolute predictors.")

# Footer
st.markdown("""
---
**📚 Metrics Guide** | Investment Analysis Portal | For educational purposes only
""")