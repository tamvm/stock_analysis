#!/usr/bin/env python3
"""
Import market-level statistics (PE ratio, market cap, earnings, revenue)
Supports VN market from Simplize API, extensible for US market in the future
"""

import requests
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

# Market configurations
MARKET_CONFIGS = {
    'vn': {
        'name': 'Vietnam Market',
        'api_url': 'https://api2.simplize.vn/api/historical/statistics/index?period=10y',
        'headers': {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'en-US,en;q=0.9',
            'origin': 'https://simplize.vn',
            'referer': 'https://simplize.vn/',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
        }
    },
    # Future: Add US market configuration
    # 'us': {
    #     'name': 'US Market',
    #     'api_url': 'TBD',
    #     'headers': {...}
    # }
}

def fetch_vn_market_statistics():
    """
    Fetch Vietnam market statistics from Simplize API
    
    Returns:
        DataFrame with columns: date, pe_ratio, total_market_cap, total_earnings, total_revenue
        or None if failed
    """
    config = MARKET_CONFIGS['vn']
    
    try:
        print(f"  Fetching data from Simplize API...")
        response = requests.get(config['api_url'], headers=config['headers'])
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' not in data or 'items' not in data['data']:
            print(f"  ⚠️  Invalid response format")
            return None
        
        items = data['data']['items']
        pe_median = data['data'].get('peMedian')
        
        print(f"  ✓ API returned {len(items)} records")
        print(f"    PE Median: {pe_median}")
        
        # Process items: [timestamp, PE, total_market_cap, total_earnings, total_revenue]
        records = []
        for item in items:
            if isinstance(item, list) and len(item) >= 5:
                timestamp = item[0]  # Unix timestamp in seconds
                pe_ratio = item[1]
                total_market_cap = item[2]
                total_earnings = item[3]
                total_revenue = item[4]
                
                records.append({
                    'date': pd.to_datetime(timestamp, unit='s'),
                    'pe_ratio': float(pe_ratio) if pe_ratio else None,
                    'total_market_cap': float(total_market_cap) if total_market_cap else None,
                    'total_earnings': float(total_earnings) if total_earnings else None,
                    'total_revenue': float(total_revenue) if total_revenue else None
                })
        
        if not records:
            print(f"  ⚠️  No valid records found")
            return None
        
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  ✓ Processed {len(df)} unique records")
        print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"    PE range: {df['pe_ratio'].min():.2f} to {df['pe_ratio'].max():.2f}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ API request failed: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error processing data: {e}")
        return None

def import_market_statistics(market_code, db_path="db/investment_data.db", clear_existing=True):
    """
    Import market statistics for a given market
    
    Args:
        market_code: Market code ('vn', 'us', etc.)
        db_path: Path to SQLite database
        clear_existing: If True, clear existing data for this market before importing
    
    Returns:
        True if successful, False otherwise
    """
    market_code = market_code.lower()
    
    if market_code not in MARKET_CONFIGS:
        print(f"✗ Unknown market code: {market_code}")
        print(f"  Available markets: {', '.join(MARKET_CONFIGS.keys())}")
        return False
    
    config = MARKET_CONFIGS[market_code]
    market_code_upper = market_code.upper()
    
    print(f"\n{'='*60}")
    print(f"Importing {config['name']} Statistics")
    print(f"{'='*60}")
    
    # Fetch data based on market
    if market_code == 'vn':
        df = fetch_vn_market_statistics()
    else:
        print(f"✗ Import function not implemented for {market_code}")
        return False
    
    if df is None or len(df) == 0:
        print(f"✗ Failed to fetch data for {market_code_upper}")
        return False
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        if clear_existing:
            print(f"  Clearing existing {market_code_upper} data...")
            conn.execute('DELETE FROM market_statistics WHERE market_code = ?', (market_code_upper,))
        
        # Prepare data for database
        df_db = df.copy()
        df_db['market_code'] = market_code_upper
        
        # Reorder columns to match table schema
        df_db = df_db[['market_code', 'date', 'pe_ratio', 'total_market_cap', 'total_earnings', 'total_revenue']]
        
        # Insert data
        print(f"  Inserting {len(df_db)} records...")
        df_db.to_sql('market_statistics', conn, if_exists='append', index=False)
        
        conn.commit()
        
        print(f"✅ Successfully imported {market_code_upper} market statistics:")
        print(f"   - Market: {config['name']}")
        print(f"   - Records: {len(df_db)}")
        print(f"   - Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"   - PE Range: {df['pe_ratio'].min():.2f} to {df['pe_ratio'].max():.2f}")
        
        # Verify import
        result = conn.execute(
            'SELECT COUNT(*) FROM market_statistics WHERE market_code = ?', 
            (market_code_upper,)
        ).fetchone()
        print(f"   - Verified: {result[0]} records in database")
        
        # Show sample data
        sample = conn.execute('''
            SELECT date, pe_ratio, total_market_cap 
            FROM market_statistics 
            WHERE market_code = ? 
            ORDER BY date DESC 
            LIMIT 5
        ''', (market_code_upper,)).fetchall()
        
        print(f"\n   Latest 5 records:")
        print(f"   {'Date':<12} {'PE Ratio':<10} {'Market Cap':<20}")
        print(f"   {'-'*42}")
        for row in sample:
            date_str = row[0][:10] if row[0] else 'N/A'
            pe_str = f"{row[1]:.2f}" if row[1] else 'N/A'
            mc_str = f"{row[2]:.2e}" if row[2] else 'N/A'
            print(f"   {date_str:<12} {pe_str:<10} {mc_str:<20}")
        
        return True
        
    except Exception as e:
        print(f"✗ Database error for {market_code_upper}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def import_all_markets(db_path="db/investment_data.db"):
    """
    Import statistics for all configured markets
    
    Args:
        db_path: Path to SQLite database
    """
    print("="*60)
    print("Market Statistics Import")
    print("="*60)
    print(f"Available markets: {len(MARKET_CONFIGS)}")
    print(f"Database: {db_path}")
    print()
    
    success_count = 0
    fail_count = 0
    
    for market_code in MARKET_CONFIGS.keys():
        if import_market_statistics(market_code, db_path):
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Import Summary")
    print("="*60)
    print(f"Total markets: {len(MARKET_CONFIGS)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    # Show database status
    if success_count > 0:
        conn = sqlite3.connect(db_path)
        
        summary = pd.read_sql_query("""
            SELECT 
                market_code,
                COUNT(*) as record_count,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                MIN(pe_ratio) as min_pe,
                MAX(pe_ratio) as max_pe,
                AVG(pe_ratio) as avg_pe
            FROM market_statistics
            GROUP BY market_code
            ORDER BY market_code
        """, conn)
        
        print("\nMarket Statistics Summary:")
        print(summary.to_string(index=False))
        
        conn.close()

def main():
    """Main function"""
    db_path = "db/investment_data.db"
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"Error: Database {db_path} not found")
        print("Please create the database first by running: python run_migrations.py")
        return
    
    if len(sys.argv) > 1:
        # Single market import mode
        market_code = sys.argv[1].lower()
        import_market_statistics(market_code, db_path)
    else:
        # Import all markets
        import_all_markets(db_path)

if __name__ == "__main__":
    main()
