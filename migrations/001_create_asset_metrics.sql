-- Migration 001: Create Asset Metrics Table
-- Stores calculated financial metrics for each asset

CREATE TABLE IF NOT EXISTS asset_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_code TEXT NOT NULL,
    calculated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Returns
    return_1y REAL,
    return_3y_annualized REAL,
    return_5y_annualized REAL,
    
    -- Risk Metrics
    std_dev_annualized REAL,
    sharpe_ratio REAL,
    max_drawdown_3y REAL,
    max_drawdown_5y REAL,
    
    -- Rolling CAGR (stored as JSON for multiple windows)
    rolling_cagr_1y TEXT,  -- JSON array of {date, cagr} objects
    rolling_cagr_3y TEXT,
    rolling_cagr_4y TEXT,
    rolling_cagr_5y TEXT,
    
    FOREIGN KEY (asset_code) REFERENCES assets(asset_code),
    UNIQUE(asset_code)  -- Only keep latest calculation per asset
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_asset_metrics_asset_code ON asset_metrics(asset_code);
CREATE INDEX IF NOT EXISTS idx_asset_metrics_calculated_at ON asset_metrics(calculated_at);
