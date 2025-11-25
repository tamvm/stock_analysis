# 📈 Investment Analysis Portal

Professional investment analysis and comparison tool for mutual funds and benchmark indices.

## Features

- **Rolling CAGR Analysis**: Compare rolling 1Y, 3Y, 4Y, and 5Y compound annual growth rates
- **Multi-Asset Comparison**: Select and toggle multiple funds and benchmarks
- **Professional Metrics**: Sharpe ratio, maximum drawdown, volatility, correlation analysis
- **Interactive Charts**: Hover tooltips, zoom/pan, date range selection with enhanced visibility
- **Benchmark Comparison**: Compare against VN-Index and S&P 500
- **Performance Summary**: 3Y, 5Y, and since-inception analysis
- **📚 Metrics Guide**: Comprehensive explanation of all financial metrics and formulas
- **Enhanced Visualization**: Improved chart colors and line thickness for better readability

## Available Assets

### 🏦 Mutual Funds
- **DCDS** (since 2004) - 3,458 records
- **DCDE** (since 2008) - 3,390 records
- **VCBFBCF** (since 2014) - 881 records
- **MAGEF** (since 2019) - 919 records
- **UVEEF** (since 2022) - 750 records
- **VCAMDF** (since 2024) - 375 records

### 📊 Benchmarks
- **VN-Index** (since 2017) - 2,220 records
- **S&P 500** (since 2017) - 2,265 records

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Process Data
```bash
python setup.py
```

### 2.1 Import Individual Fund/Stock Data
Use the dedicated import scripts to add new data:

#### Fund Data (JSON format)
```bash
# Import fund data
python import_fund_data.py data/vesaf-20251120.txt VESAF
python import_fund_data.py data/vemeef-20251120.txt VEMEEF

# List all assets in database
python import_fund_data.py --list-assets

# View help
python import_fund_data.py --help
```

#### Stock/ETF Data (CSV or JSON format)
```bash
# Import stock/ETF data with custom name
python import_stock_data.py data/vti-112025.csv "Vanguard Total Stock Market ETF"
python import_stock_data.py data/qqq-112025.csv "Invesco QQQ ETF"

# Import using filename as asset code
python import_stock_data.py data/spy-112025.csv

# Import JSON fund data (alternative to import_fund_data.py)
python import_stock_data.py data/dcbf-20251120.txt "Dragon Capital Balanced Fund"
```

### 3. Launch Dashboard
```bash
streamlit run dashboard.py
```

### 4. Open Browser
Navigate to `http://localhost:8501`

## Dashboard Features

### 📊 Multi-Page Interface
- **Main Dashboard**: Investment analysis with interactive charts
- **📚 Metrics Guide**: Comprehensive explanation of all financial metrics with formulas and examples

### 🎯 Asset Selection
- Multi-select dropdown for funds and benchmarks
- Toggle assets on/off for focused analysis
- No asset grouping - simple flat selection

### 📊 Rolling CAGR Chart
- **Default**: 4-year rolling CAGR
- **Options**: 1Y, 3Y, 4Y, 5Y periods
- Interactive timeline with hover details
- Multiple asset overlay comparison

### 📈 Performance Analysis
- **Cumulative Returns**: Normalized to 100% start point
- **Risk-Return Scatter**: Volatility vs returns positioning
- **Correlation Matrix**: Asset relationship heatmap
- **Drawdown Analysis**: Maximum decline periods

### 📚 Metrics Guide Page
- **Performance Metrics**: Total return, CAGR, rolling returns explained with formulas
- **Risk Metrics**: Volatility, max drawdown, Sharpe ratio with practical examples
- **Correlation Analysis**: Diversification principles and portfolio construction
- **Technical Analysis**: Chart interpretation and investment guidelines
- **Practical Guidelines**: Conservative, balanced, and aggressive portfolio targets

### 🔍 Interactive Features
- **Hover tooltips** with exact values
- **Zoom and pan** capabilities
- **Date range selectors** (optional)
- **Performance period selection**: 3Y, 5Y, inception
- **Enhanced Chart Colors**: Improved visibility with thicker lines and better color palette

## Key Metrics Calculated

### Performance Metrics
- **Total Return**: Absolute percentage gain/loss
- **Annualized Return (CAGR)**: Compound annual growth rate
- **Rolling Returns**: Moving window analysis
- **Excess Returns**: Performance above benchmarks

### Risk Metrics
- **Volatility**: Standard deviation of returns (annualized)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns (2% risk-free rate)
- **Sortino Ratio**: Downside risk-adjusted returns

### Correlation Analysis
- **Correlation Matrix**: Asset relationships
- **Rolling Correlations**: Time-varying dependencies

## Technical Architecture

### Data Processing
- **SQLite Database**: Unified storage format in `/db` folder
- **JSON Fund Data**: Automatic parsing of NAV history
- **CSV Benchmark Data**: Price history processing
- **Data Cleaning**: Duplicate removal and date standardization

### Calculation Engine
- **Pandas**: Time series analysis and calculations
- **NumPy**: Statistical computations
- **SciPy**: Advanced statistical functions

### Visualization
- **Streamlit**: Interactive web dashboard
- **Plotly**: Professional interactive charts
- **Responsive Design**: Multi-column layouts

## File Structure

```
stock_analysis/
├── data/                           # Source data files
│   ├── dcds-20251120.txt          # Fund NAV history (JSON)
│   ├── dcde-20251120.txt
│   ├── magef-20251120.txt
│   ├── uveef-20251120.txt
│   ├── vcamdf-20251120.txt
│   ├── vcbfbcf-20251120.txt
│   └── references/
│       ├── vnindex-2017.csv       # VN-Index price history
│       └── voo-2017.csv           # S&P 500 price history
├── db/                            # Database storage
│   └── investment_data.db         # SQLite database
├── pages/                         # Streamlit multi-page structure
│   └── 1_📚_Metrics_Guide.py     # Financial metrics explanation page
├── dashboard.py                   # Main Streamlit application
├── data_processor.py              # Data loading and processing
├── metrics_calculator.py          # Financial calculations
├── setup.py                      # Initial data processing
├── import_fund_data.py            # Individual fund data import script
├── import_stock_data.py           # Stock/ETF data import script (CSV/JSON)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Data Sources

### US Stocks Data
- **Source**: [NASDAQ Historical Data](https://www.nasdaq.com/market-activity/etf/vti/historical?page=1&rows_per_page=10&timeline=y10)
- **Coverage**: ETF historical data with 10-year timeline
- **Format**: CSV download available

### Vietnam Stocks Data
- **Source**: [FMarket VN Fund Data](https://fmarket.vn/quy/vcbfbcf)
- **Coverage**: Vietnamese mutual fund data
- **Format**: JSON fund NAV history

## Data Update Process

### Adding New Assets
1. Add new data file to `/data` folder
   - **Funds**: JSON format (use `import_fund_data.py`)
   - **Stocks/ETFs**: CSV format (use `import_stock_data.py`)
2. Import using appropriate script:
   - `python import_fund_data.py data/newfund-20251120.txt NEWFUND`
   - `python import_stock_data.py data/newstock-112025.csv "New Stock Name"`
3. Verify with: `python import_fund_data.py --list-assets`

### Bulk Updates (Future)
1. Add new data files to `/data` folder
2. Run `python setup.py` to reprocess all
3. Database automatically updates with new records

### Current Status
- **Static Data**: Working with current dataset
- **Real-time Fetching**: Planned for future implementation
- **Data Range**: 2004-2025 historical coverage

## Investment Decision Support

### Portfolio Optimization
- **Conservative**: Low volatility, steady returns (DCDS, DCDE)
- **Growth**: Higher risk, higher potential (newer funds)
- **Balanced**: Optimized risk-return ratio
- **Benchmark Tracking**: Index comparison analysis

### Analysis Workflow
1. **Select Assets**: Choose funds for comparison
2. **Set Time Period**: 4Y rolling CAGR (default)
3. **Review Performance**: Check summary metrics
4. **Analyze Risk**: Review drawdowns and volatility
5. **Compare Correlations**: Understand diversification
6. **Make Decision**: Based on risk tolerance and goals

## Professional Features

### Chart Types
- **Rolling CAGR**: Like professional investment platforms
- **Cumulative Performance**: Total return visualization
- **Risk-Return Efficiency**: Portfolio positioning
- **Drawdown Recovery**: Risk assessment
- **Correlation Dynamics**: Diversification analysis

### Metrics Standards
- **Annualized Returns**: Industry standard CAGR
- **Sharpe Ratios**: Risk-adjusted performance
- **Maximum Drawdown**: Worst-case scenario analysis
- **Volatility**: Standard deviation metrics

## Notes

- **Educational Purpose**: For investment research and education
- **Weekly Data Updates**: Manual process currently
- **No Export Features**: Dashboard-only analysis
- **Professional Grade**: Industry-standard calculations

---

**Investment Analysis Portal** | Built with Streamlit & Plotly | Professional Investment Analysis Tool