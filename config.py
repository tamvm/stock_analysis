"""
Configuration file for Investment Analysis Portal
Centralized constants and settings
"""

# Financial Constants
RISK_FREE_RATE = 0.045  # 4.5% - Current 10-year US Treasury yield (Dec 2024)

# Trading days per year by asset type
# Used for annualizing volatility and returns
TRADING_DAYS = {
    'fund': 252,       # Traditional mutual funds (weekday trading)
    'etf': 252,        # Exchange-traded funds (weekday trading)
    'benchmark': 252,  # Stock market indices (weekday trading)
    'stock': 252,      # Individual stocks (weekday trading)
    'crypto': 365,     # Cryptocurrencies (24/7 trading)
}

def get_trading_days(asset_type):
    """
    Get appropriate trading days for asset type
    
    Args:
        asset_type (str): Type of asset ('fund', 'etf', 'benchmark', 'stock', 'crypto')
        
    Returns:
        int: Number of trading days per year (252 or 365)
    """
    return TRADING_DAYS.get(asset_type, 252)  # Default to 252 for unknown types

# Database Configuration
DB_PATH = 'db/investment_data.db'
MIGRATIONS_PATH = 'migrations/'

# Calculation Settings
MIN_YEARS_FOR_3Y_METRICS = 3  # Minimum years of data required for 3-year metrics
MIN_YEARS_FOR_5Y_METRICS = 5  # Minimum years of data required for 5-year metrics
MIN_YEARS_FOR_1Y_METRICS = 1  # Minimum years of data required for 1-year metrics

# Rolling CAGR Windows (in years)
ROLLING_WINDOWS = [1, 3, 4, 5]

# Asset Presets for Quick Selection
# Users can add custom presets by editing this dictionary
# Format: 'Preset Name': ['ASSET1', 'ASSET2', ...]
ASSET_PRESETS = {
    'All Funds': [],  # Will be populated dynamically with all fund assets
    'All ETFs': [],   # Will be populated dynamically with all ETF assets
    'All Benchmarks': [],  # Will be populated dynamically with all benchmark assets
    
    # Custom presets - edit these for your presentation needs
    'My Presentation': ['VCBFBCF', 'VNINDEX', 'SP500'],
    'Top Performers': ['UVEEF', 'DCDS', 'VNINDEX'],
    'Conservative Mix': ['DCBF', 'VCBFTBF', 'SP500'],
}

# Chart Settings
ROLLING_WINDOW_DAYS = 90  # Rolling window for volatility and Sharpe ratio charts (90 days recommended)
