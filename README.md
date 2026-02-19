# 📈 Investment Analysis Portal

Professional investment analysis and comparison tool for mutual funds, stocks, ETFs, cryptocurrencies, and benchmark indices.

## Features

- **Multi-Asset Support**: Vietnamese funds, US stocks/ETFs, cryptocurrencies (BTC, ETH), and market indices
- **Rolling CAGR Analysis**: Compare rolling 1Y, 3Y, 4Y, and 5Y compound annual growth rates
- **Multi-Asset Comparison**: Select and toggle multiple assets across different types
- **Professional Metrics**: Sharpe ratio, maximum drawdown, volatility, correlation analysis
- **Interactive Charts**: Hover tooltips, zoom/pan, date range selection with enhanced visibility
- **Benchmark Comparison**: Compare against VN-Index, VN30, and S&P 500
- **Performance Summary**: 3Y, 5Y, and since-inception analysis
- **📚 Metrics Guide**: Comprehensive explanation of all financial metrics and formulas
- **Enhanced Visualization**: Improved chart colors and line thickness for better readability
- **Cryptocurrency Analytics**: 24/7 trading data with specialized volatility calculations

## Available Assets

### 🏦 Vietnamese Mutual Funds (11 funds)
- **DCDS** (since 2004-05-20) - 3,486 records
- **DCDE** (since 2008-02-29) - 3,418 records
- **DCBF** (since 2013-06-10) - 1,038 records
- **VCBFTBF** (since 2013-12-26) - 974 records
- **VCBFBCF** (since 2014-08-27) - 905 records
- **SSISCA** (since 2014-09-26) - 2,024 records
- **VESAF** (since 2017-04-25) - 1,329 records
- **MAGEF** (since 2019-07-23) - 947 records
- **UVEEF** (since 2022-11-08) - 778 records
- **VEMEEF** (since 2023-05-09) - 650 records
- **VCAMDF** (since 2024-03-18) - 403 records

### 📊 Vietnamese Indices (2 indices)
- **VNINDEX** - VN-Index (since 2000-06-30) - 306 records
- **VN30** - VN30 Index (since 2012-01-31) - 167 records

### 🇺🇸 US ETFs (3 ETFs)
- **QQQ** - Invesco QQQ Trust (since 2015-12-30) - 2,513 records
- **VOO** - Vanguard S&P 500 ETF (since 2015-12-30) - 2,513 records
- **VTI** - Vanguard Total Stock Market ETF (since 2015-12-30) - 2,513 records

### 📈 US Stocks (4 stocks)
- **AMZN** - Amazon (since 2015-12-30) - 2,513 records
- **GOOG** - Google (since 2015-12-30) - 2,513 records
- **META** - Meta Platforms (since 2015-12-30) - 2,513 records
- **TSLA** - Tesla (since 2015-12-30) - 2,513 records

### ₿ Cryptocurrencies (2 assets)
- **BTC** - Bitcoin (since 2014-09-17) - 4,121 records
- **ETH** - Ethereum (since 2017-11-09) - 2,972 records

### 📊 Benchmarks (1 index)
- **SP500** - S&P 500 Index (since 2017-01-03) - 2,265 records

**Total: 23 assets** across 6 asset types with comprehensive historical price data

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Database
```bash
# Run database migrations
python run_migrations.py

# Check migration status
python run_migrations.py --status
```

### 3. Process Data
```bash
python setup.py
```

### 3.1 Import Individual Fund/Stock Data

#### Import US Stocks/ETFs (CSV files)
Use this script for US stocks and ETFs from the `data/us/` folder:

```bash
# Batch import all CSV files from data/us/ folder
python import_us_data.py

# Import single US stock/ETF (CSV format)
python import_us_data.py data/us/HistoricalData_1767040077932-qqq.csv

# Import with custom asset name
python import_us_data.py data/us/HistoricalData_1767040130966-vti.csv "Vanguard Total Stock Market ETF"
```

#### Import Vietnamese Funds (API)
Use the dedicated VN funds import script to fetch data directly from the fmarket API:

```bash
# Import all VN assets (funds + indices) - recommended
python import_vn_funds.py

# Import single fund by asset code
python import_vn_funds.py vesaf
python import_vn_funds.py dcds

# Import single index
python import_vn_funds.py vnindex
python import_vn_funds.py vn30
```

**Supported VN Funds (fmarket API):**
- `vesaf`, `dcbf`, `dcde`, `dcds`, `magef`, `ssisca`, `uveef`, `vcamdf`, `vcbfbcf`, `vcbftbf`, `vemeef`

**Supported VN Indices (Simplize API):**
- `vnindex` - VN-Index (since 2000, 306 monthly records)
- `vn30` - VN30 Index (since 2012, 167 monthly records)

#### Import VN Fund Metadata (API)
Use this script to fetch and store detailed fund information including fees, holdings, and sector allocation:

```bash
# Import metadata for all VN funds - recommended
python import_vn_fund_metadata.py

# Import metadata for a single fund
python import_vn_fund_metadata.py dcds
python import_vn_fund_metadata.py vcbfbcf
```

**What gets imported:**
- **Fund Information**: Name, issuer, asset type, total assets, management fees
- **Fee Structure**: Purchase fees and redemption fees (by holding period)
- **Top Holdings**: Top 10 stocks/bonds with sector, % allocation, and current prices
- **Sector Allocation**: Percentage breakdown by industry sector

**Note:** This is separate from NAV/price data import. Run `import_vn_funds.py` for price history and `import_vn_fund_metadata.py` for fund details.

#### Import Cryptocurrencies (Yahoo Finance API)
Use the dedicated crypto import script to fetch data directly from Yahoo Finance using the yfinance library:

```bash
# Import all cryptocurrencies - recommended
python import_crypto_data.py

# Import single cryptocurrency by asset code
python import_crypto_data.py btc
python import_crypto_data.py eth
```

**Supported Cryptocurrencies (Yahoo Finance API):**
- `btc` - Bitcoin (BTC-USD, since 2014-09-17, ~4,121 records)
- `eth` - Ethereum (ETH-USD, since 2017-11-09, ~2,972 records)

**Key Points:**
- `import_us_data.py`: For US stocks/ETFs (CSV files from `data/us/` folder)
- `import_vn_funds.py`: For Vietnamese funds and indices (API data)
  - VN Funds: fmarket API with `asset_type='vn_fund'`
  - VN Indices: Simplize API with `asset_type='vn_index'`
- `import_crypto_data.py`: For cryptocurrencies (Yahoo Finance API)
  - Uses yfinance library with `asset_type='crypto'`
  - Daily price data with 365 trading days/year for volatility calculations
- All scripts automatically detect asset codes and override existing data

### 4. Calculate Metrics (Optional)
```bash
# Calculate all metrics via command line
python metrics_job_manager.py

# Or use the UI button in the Assets List page
```

### 5. Launch Dashboard
```bash
streamlit run dashboard.py
```

### 6. Open Browser
Navigate to `http://localhost:8501`

---

## 🐳 Docker Deployment (Server)

Use Docker Compose to run the app on a server. The SQLite database and data directory are mounted as volumes so they persist across container restarts.

### Prerequisites
- Docker & Docker Compose installed on the server
- The `db/investment_data.db` file present in the project directory

### 1. Clone the repo on the server
```bash
git clone <your-repo-url> stock_analysis
cd stock_analysis
```

> **Note:** The `db/` directory with `investment_data.db` is **not** committed to git (it's in `.gitignore`). Copy it to the server manually:
> ```bash
> scp -r ./db user@your-server:/path/to/stock_analysis/
> ```

### 2. Build and start
```bash
docker compose up -d --build
```

The dashboard will be available at `http://localhost:8501`.

### 3. View logs
```bash
docker compose logs -f
```

### 4. Stop
```bash
docker compose down
```

### 5. Run data import scripts inside the container
Since imports are manual, exec into the running container:

```bash
# Update VN fund data
docker compose exec stock_analysis python import_vn_funds.py

# Update US data
docker compose exec stock_analysis python import_us_data.py

# Update crypto data
docker compose exec stock_analysis python import_crypto_data.py

# Recalculate metrics
docker compose exec stock_analysis python metrics_job_manager.py
```

### 6. Cloudflare Tunnel (cloudflared)

Add the following ingress rule to your existing `~/.cloudflared/config.yml` **before** the catch-all `http_status:404` line:

```yaml
tunnel: n8n
credentials-file: /home/pi/.cloudflared/4c1c8e3f-0744-4053-b239-efcc17fe0444.json

ingress:
  - hostname: openclaw-x.kenchange.com
    service: http://localhost:3000
  - hostname: n8n.kenchange.com
    service: http://localhost:5678
  # --- Add this block ---
  - hostname: fund_analysis.kenchange.com
    service: http://localhost:8501
  # ----------------------
  - service: http_status:404
```

Then restart the tunnel:
```bash
sudo systemctl restart cloudflared
```

The dashboard will be accessible at `https://fund_analysis.kenchange.com`.

---

## Dashboard Features

### 📊 Multi-Page Interface
- **Main Dashboard**: Investment analysis with interactive charts
- **📚 Metrics Guide**: Comprehensive explanation of all financial metrics with formulas and examples
- **🏆 Top Performers**: Yearly top 3 assets with highest annual growth, with VNINDEX reference and asset type filtering

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

## Asset Metrics Calculation System

### Overview
The portal includes an automated metrics calculation system that computes comprehensive financial performance indicators for all assets. Metrics are calculated on-demand and stored in the database for quick access.

### Calculated Metrics

#### Performance Metrics
- **1-Year Return**: Absolute return over the past year
- **3-Year Annualized Return (p.a.)**: Compound annual growth rate over 3 years
- **5-Year Annualized Return (p.a.)**: Compound annual growth rate over 5 years
- **Rolling CAGR**: Moving window analysis for 1Y, 3Y, 4Y, and 5Y periods

#### Risk Metrics
- **Annualized Standard Deviation**: Volatility measure (annualized)
  - Traditional assets (stocks, ETFs, funds): Based on 252 trading days/year
  - Crypto assets: Based on 365 trading days/year (24/7 trading)
- **Sharpe Ratio**: Risk-adjusted returns using 4.5% risk-free rate (current 10-year US Treasury yield)
- **Maximum Drawdown (3Y)**: Largest peak-to-trough decline over 3 years
- **Maximum Drawdown (5Y)**: Largest peak-to-trough decline over 5 years

### How to Calculate Metrics

#### Via UI (Recommended)
1. Navigate to the **Assets List** page
2. Click the **"🔄 Calculate Metrics"** button
3. Watch the progress bar as metrics are calculated for each asset
4. View calculated metrics in asset cards and comparison table

#### Via Command Line
```bash
python metrics_job_manager.py
```

### Database Migrations

The system uses a simple SQL-based migration system to manage database schema changes.

#### Running Migrations
```bash
# Run all pending migrations
python run_migrations.py

# Check migration status
python run_migrations.py --status
```

#### Migration Files
- `000_initial_schema.sql` - Documents existing schema
- `001_create_asset_metrics.sql` - Creates asset_metrics table
- `002_create_calculation_runs.sql` - Creates calculation_runs tracking table

### Configuration

Financial constants are defined in `config.py`:

```python
RISK_FREE_RATE = 0.045  # 4.5% - Current 10-year US Treasury yield

TRADING_DAYS = {
    'fund': 252,       # Traditional mutual funds
    'etf': 252,        # Exchange-traded funds
    'benchmark': 252,  # Stock market indices
    'crypto': 365,     # Cryptocurrencies (24/7 trading)
    'stock': 252       # Individual stocks
}
```

### Background Processing

Metrics calculation runs asynchronously in a background thread:
- **Progress Tracking**: Real-time updates via progress bar
- **Error Handling**: Individual asset failures don't stop the entire job
- **Persistence**: Results stored in `asset_metrics` table
- **History**: Calculation runs tracked in `calculation_runs` table

### Data Requirements

- **1-Year Metrics**: Requires at least 1 year of price data
- **3-Year Metrics**: Requires at least 3 years of price data
- **5-Year Metrics**: Requires at least 5 years of price data
- **Insufficient Data**: Metrics are left as NULL if data is unavailable

## Key Metrics Calculated (Legacy)

### Performance Metrics
- **Total Return**: Absolute percentage gain/loss
- **Annualized Return (CAGR)**: Compound annual growth rate
- **Rolling Returns**: Moving window analysis
- **Excess Returns**: Performance above benchmarks

### Risk Metrics
- **Volatility**: Standard deviation of returns (annualized)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns (4.5% risk-free rate)
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
├── data/                           # Vietnamese fund data
│   ├── dcds-20251120.txt          # Fund NAV history (JSON)
│   ├── dcde-20251120.txt
│   ├── magef-20251120.txt
│   ├── uveef-20251120.txt
│   ├── vcamdf-20251120.txt
│   ├── vcbfbcf-20251120.txt
│   └── references/
│       ├── vnindex-2017.csv       # VN-Index price history
│       └── voo-2017.csv           # S&P 500 price history
├── data/us/                       # US stock/ETF data
│   ├── HistoricalData_*-amzn.csv  # Amazon stock price history
│   ├── HistoricalData_*-googl.csv # Google stock price history
│   ├── HistoricalData_*-meta.csv  # Meta stock price history
│   ├── HistoricalData_*-qqq.csv   # QQQ ETF price history
│   ├── HistoricalData_*-tsla.csv  # Tesla stock price history
│   ├── HistoricalData_*-voo.csv   # VOO ETF price history
│   └── HistoricalData_*-vti.csv   # VTI ETF price history
├── db/                            # Database storage
│   └── investment_data.db         # SQLite database
├── pages/                         # Streamlit multi-page structure
│   └── 1_📚_Metrics_Guide.py     # Financial metrics explanation page
├── dashboard.py                   # Main Streamlit application
├── data_processor.py              # Data loading and processing
├── metrics_calculator.py          # Financial calculations
├── setup.py                      # Initial data processing
├── import_us_data.py              # Unified data import script (CSV/JSON)
├── import_vn_funds.py             # VN funds API import script
├── import_crypto_data.py          # Cryptocurrency import script (Yahoo Finance)
├── import_nasdaq_data.py          # Legacy NASDAQ import script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Data Sources

### US Stocks/ETFs Data
- **Source**: [NASDAQ Historical Data](https://www.nasdaq.com/market-activity/etf/vti/historical?page=1&rows_per_page=10&timeline=y10)
- **Format**: CSV files with pattern `HistoricalData_<timestamp>-<symbol>.csv`
- **Location**: `data/us/` folder
- **Assets**: AMZN, GOOGL, META, QQQ, TSLA, VOO, VTI

### Vietnamese Fund Data
- **Source**: [FMarket VN Fund Data](https://fmarket.vn/quy/vcbfbcf)
- **Format**: JSON files with `.txt` extension
- **Location**: `data/` folder
- **Method**: Inspect network requests to download the data

### Cryptocurrency Data
- **Source**: Yahoo Finance API via yfinance library - [web link](https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD?events=capitalGain%7Cdiv%7Csplit&formatted=true&includeAdjustedClose=true&interval=1d&period1=1410912000&period2=1767061845&symbol=BTC-USD&userYfid=true&lang=en-AU&region=AU)
- **Format**: API data (no local files needed)
- **Assets**: BTC (Bitcoin), ETH (Ethereum)
- **Method**: Automated fetch using `import_crypto_data.py`
- **Update**: Run script to fetch latest data from Yahoo Finance

### Market PE Data (US)
- **Source**: [SimplyWall.St US Market](https://simplywall.st/markets/us)
- **Format**: JSON file with PE ratio timeseries data
- **Location**: `data/us/simplywall_pe.json`
- **Method**: Manual update via browser DevTools
- **Update Process**:
  1. Visit https://simplywall.st/markets/us
  2. Open browser DevTools (F12) → Network tab
  3. Filter by "graphql" requests
  4. Look for response containing `IndustryTimeseries` data
  5. Copy the JSON response
  6. Save to `data/us/simplywall_pe.json`
  7. Run `python import_market_pe.py us` to import
- **Data Fields**: `absolutePE` (mapped to `pe_ratio`), `marketCap`, `earnings`, `revenue`

## Data Update Process

### Adding New Assets
1. Add new data file to appropriate folder:
   - **Vietnamese Funds**: JSON files to `data/` folder
   - **US Stocks/ETFs**: CSV files to `data/us/` folder
2. Import using unified script:
   ```bash
   # Single file
   python import_us_data.py data/newfund-20251120.txt "New Fund Name"
   python import_us_data.py data/us/HistoricalData_*-spy.csv
   
   # Batch import all folders
   python import_us_data.py
   ```
3. Verify import in the dashboard Assets List page

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