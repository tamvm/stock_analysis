-- Migration 004: Market Statistics Table
-- Store market-level statistics (PE ratio, market cap, earnings, revenue)
-- Supports multiple markets (VN, US, etc.)

CREATE TABLE IF NOT EXISTS market_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_code TEXT NOT NULL,
    date DATE NOT NULL,
    pe_ratio REAL,
    total_market_cap REAL,
    total_earnings REAL,
    total_revenue REAL,
    UNIQUE(market_code, date)
);

CREATE INDEX IF NOT EXISTS idx_market_statistics_market_date 
    ON market_statistics(market_code, date);
