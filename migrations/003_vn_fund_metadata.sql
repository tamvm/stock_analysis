-- Migration 003: VN Fund Metadata Tables
-- Creates tables to store Vietnamese fund metadata from fmarket API

-- Fund basic information and metadata
CREATE TABLE IF NOT EXISTS fund_info (
    product_id INTEGER PRIMARY KEY,
    asset_code TEXT NOT NULL,
    fund_name TEXT,
    fund_short_name TEXT,
    fund_code TEXT,
    issuer_name TEXT,
    issuer_short_name TEXT,
    fund_asset_type TEXT,  -- e.g., "Quỹ cổ phiếu", "Quỹ trái phiếu"
    fund_asset_code TEXT,  -- e.g., "STOCK", "BOND"
    description TEXT,
    total_assets_value REAL,  -- in VND
    total_assets_value_str TEXT,  -- formatted string, e.g., "5774.0 tỷ"
    total_assets_report_date DATE,
    management_fee REAL,
    performance_fee REAL,
    nav REAL,  -- current NAV
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_code) REFERENCES assets(asset_code)
);

-- Fund fee structure (purchase and redemption fees)
CREATE TABLE IF NOT EXISTS fund_fees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    asset_code TEXT NOT NULL,
    fee_type TEXT NOT NULL,  -- 'BUY' or 'SELL'
    fee_unit_type TEXT,  -- 'MONEY' for purchase, 'MONTH' for redemption
    begin_volume REAL,  -- minimum value (amount for BUY, months for SELL)
    end_volume REAL,  -- maximum value (NULL for unlimited)
    fee_percentage REAL,  -- fee percentage
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES fund_info(product_id),
    FOREIGN KEY (asset_code) REFERENCES assets(asset_code)
);

-- Fund top holdings (stocks/bonds)
CREATE TABLE IF NOT EXISTS fund_holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    asset_code TEXT NOT NULL,
    stock_code TEXT NOT NULL,  -- ticker symbol
    industry TEXT,  -- sector name
    holding_type TEXT,  -- 'STOCK' or 'BOND'
    net_asset_percent REAL,  -- percentage of net asset value
    price REAL,  -- current price
    price_change REAL,  -- price change from previous
    price_change_percent REAL,  -- price change percentage
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES fund_info(product_id),
    FOREIGN KEY (asset_code) REFERENCES assets(asset_code)
);

-- Fund sector/industry allocation
CREATE TABLE IF NOT EXISTS fund_sector_allocation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    asset_code TEXT NOT NULL,
    industry TEXT NOT NULL,  -- sector name
    asset_percent REAL,  -- percentage of assets
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES fund_info(product_id),
    FOREIGN KEY (asset_code) REFERENCES assets(asset_code)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_fund_info_asset_code ON fund_info(asset_code);
CREATE INDEX IF NOT EXISTS idx_fund_fees_product_id ON fund_fees(product_id);
CREATE INDEX IF NOT EXISTS idx_fund_fees_asset_code ON fund_fees(asset_code);
CREATE INDEX IF NOT EXISTS idx_fund_holdings_product_id ON fund_holdings(product_id);
CREATE INDEX IF NOT EXISTS idx_fund_holdings_asset_code ON fund_holdings(asset_code);
CREATE INDEX IF NOT EXISTS idx_fund_sector_allocation_product_id ON fund_sector_allocation(product_id);
CREATE INDEX IF NOT EXISTS idx_fund_sector_allocation_asset_code ON fund_sector_allocation(asset_code);
