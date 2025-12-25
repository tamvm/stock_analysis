-- Migration 000: Initial Schema
-- This documents the existing database schema for reference
-- Status: Already applied (existing tables)

CREATE TABLE IF NOT EXISTS price_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_code TEXT NOT NULL,
    asset_type TEXT NOT NULL,  -- 'fund', 'benchmark', 'etf', 'crypto'
    date DATE NOT NULL,
    price REAL NOT NULL,
    product_id INTEGER,
    UNIQUE(asset_code, date)
);

CREATE TABLE IF NOT EXISTS assets (
    asset_code TEXT PRIMARY KEY,
    asset_name TEXT,
    asset_type TEXT,  -- 'fund', 'benchmark', 'etf', 'crypto'
    inception_date DATE,
    last_update DATE,
    description TEXT,
    top_holdings TEXT,
    investment_strategy TEXT,
    sector_allocation TEXT,
    fund_size TEXT
);
