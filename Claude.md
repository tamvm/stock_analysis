# Investment Analysis Portal - Technical Stack

## Overview
Professional investment analysis and comparison tool for mutual funds, ETFs, and benchmark indices with automated metrics calculation.

## Technology Stack

### Backend
- **Python 3.x** - Core programming language
- **SQLite 3** - Database for storing price data, assets, and calculated metrics
- **Pandas** - Time series analysis and data manipulation
- **NumPy** - Statistical computations and numerical operations

### Frontend
- **Streamlit** - Interactive web dashboard framework
- **Plotly** - Professional interactive charts and visualizations

### Database Management
- **Custom SQL-based migrations** - Simple numbered SQL files for schema management
- **Migration runner** (`run_migrations.py`) - Automated migration execution and tracking

### Background Processing
- **Python threading** - Asynchronous metrics calculation
- **Progress tracking** - Real-time UI updates during long-running operations

### Financial Calculations
- **Custom metrics calculator** (`calculate_metrics.py`) - Comprehensive financial metrics
- **Asset type-aware calculations** - Different annualization factors for traditional vs. crypto assets
- **Risk-free rate**: 4.5% (current 10-year US Treasury yield)

## Project Structure

```
stock_analysis/
├── migrations/                     # Database migration files
│   ├── 000_initial_schema.sql     # Existing schema documentation
│   ├── 001_create_asset_metrics.sql
│   └── 002_create_calculation_runs.sql
├── pages/                          # Streamlit multi-page structure
│   ├── 1_📚_Metrics_Guide.py
│   ├── 2_📊_Asset_Detail.py
│   ├── 3_🔍_Advanced_Filter.py
│   └── 4_📋_Assets_List.py        # Metrics calculation UI
├── data/                           # Source data files
├── db/                             # SQLite database
│   └── investment_data.db
├── config.py                       # Configuration and constants
├── dashboard.py                    # Main Streamlit application
├── data_processor.py               # Data loading and processing
├── metrics_calculator.py           # Legacy metrics calculations
├── calculate_metrics.py            # New metrics calculation engine
├── metrics_job_manager.py          # Background job management
├── run_migrations.py               # Database migration runner
├── import_fund_data.py             # Fund data import script
├── import_stock_data.py            # Stock/ETF data import script
├── setup.py                        # Initial data processing
└── requirements.txt                # Python dependencies
```

## Database Schema

### Core Tables
- **`assets`** - Asset metadata (code, name, type, inception date, etc.)
- **`price_data`** - Historical price data for all assets
- **`asset_metrics`** - Calculated financial metrics (returns, Sharpe ratio, drawdowns, etc.)
- **`calculation_runs`** - Tracking table for metrics calculation jobs
- **`schema_migrations`** - Applied migrations tracking

### Asset Types
- `vn_fund` - Vietnamese mutual funds
- `us_etf` - US Exchange-traded funds
- `us_stock` - US Individual stocks
- `benchmark` - Market indices (SP500)
- `vn_index` - Vietnamese indices (VNINDEX, VN30)
- `crypto` - Cryptocurrencies (24/7 trading)

### Datetime Standardization

**CRITICAL**: All datetime values in the database MUST be stored as **timezone-naive** to prevent comparison errors when mixing assets from different sources.

#### Standard Format
```
YYYY-MM-DD HH:MM:SS
Example: 2025-12-30 00:00:00
```

**NOT**: `2025-12-30 00:00:00+00:00` (timezone-aware)

#### Implementation in Import Scripts

All import scripts (`import_crypto_data.py`, `import_us_data.py`, `import_vn_funds.py`) implement timezone normalization:

```python
# After parsing datetime
date = pd.to_datetime(date_value)

# Normalize to timezone-naive
if hasattr(date, 'tz') and date.tz is not None:
    date = date.tz_localize(None)
```

#### Why This Matters
- Different data sources may return timezone-aware or timezone-naive dates
- yfinance (crypto data) returns timezone-aware dates
- Simplize API (VN indices) returns timezone-naive dates
- fmarket API (VN funds) returns timezone-naive dates
- Mixing timezone-aware and timezone-naive dates causes Python comparison errors
- Standardizing to timezone-naive ensures consistent behavior across all calculations

#### Verification
Check datetime format in database:
```bash
sqlite3 db/investment_data.db "SELECT date FROM price_data WHERE asset_code = 'BTC' LIMIT 3;"
# Should show: 2025-12-28 00:00:00 (no +00:00)
```

## Key Features

### Automated Metrics Calculation
- **On-demand calculation** via UI button or command line
- **Background processing** with progress tracking
- **Per-asset timestamps** for calculation tracking
- **Global run history** for audit trail

### Calculated Metrics
- 1-year, 3-year, 5-year annualized returns
- Annualized standard deviation (volatility)
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown (3-year and 5-year)
- Rolling CAGR (1Y, 3Y, 4Y, 5Y windows)

### Trading Days Adjustment
- **Traditional assets** (funds, ETFs, stocks, benchmarks): 252 trading days/year
- **Crypto assets**: 365 trading days/year (24/7 trading)
- Automatic detection based on asset type

### Migration System
- **Simple SQL files** - Easy to read and modify
- **Numbered migrations** - Clear execution order
- **Idempotent** - Safe to re-run
- **Status tracking** - Know which migrations are applied

## Development Workflow

### Adding New Assets
1. Add data file to `/data` folder
2. Run import script: `python import_fund_data.py` or `python import_stock_data.py`
3. Calculate metrics via UI or `python metrics_job_manager.py`

### Database Changes
1. Create new migration file: `migrations/XXX_description.sql`
2. Run migrations: `python run_migrations.py`
3. Verify: `python run_migrations.py --status`

### Updating Financial Constants
Edit `config.py`:
- `RISK_FREE_RATE` - Current risk-free rate
- `TRADING_DAYS` - Trading days by asset type

## Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
python run_migrations.py

# Import data
python setup.py

# Calculate metrics
python metrics_job_manager.py

# Start the server manually when ready
# streamlit run dashboard.py
```

### Production (Railway)
- Configured via `railway.toml` and `Procfile`
- Automatic deployment on git push
- See `RAILWAY_DEPLOYMENT.md` for details

## Performance Considerations

- **Caching**: Streamlit `@st.cache_resource` for expensive operations
- **Background jobs**: Threading for non-blocking UI
- **Database indexes**: Optimized queries on asset_code and dates
- **Lazy loading**: Metrics calculated on-demand, not on every page load

## Future Enhancements

- Real-time data fetching from APIs
- Additional metrics (Sortino ratio, Calmar ratio, etc.)
- Portfolio optimization tools
- Export functionality (CSV, Excel)
- Multi-currency support
- User authentication and saved portfolios
