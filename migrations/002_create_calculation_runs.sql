-- Migration 002: Create Calculation Runs Table
-- Tracks global calculation job execution history

CREATE TABLE IF NOT EXISTS calculation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT NOT NULL,  -- 'running', 'completed', 'failed'
    assets_processed INTEGER DEFAULT 0,
    assets_total INTEGER DEFAULT 0,
    error_message TEXT
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_calculation_runs_status ON calculation_runs(status);
CREATE INDEX IF NOT EXISTS idx_calculation_runs_started_at ON calculation_runs(started_at);
