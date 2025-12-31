#!/usr/bin/env python3

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
import time

# Cryptocurrency asset code to Yahoo Finance symbol mapping
CRYPTO_SYMBOLS = {
    'btc': {
        'symbol': 'BTC-USD',
        'name': 'Bitcoin',
        'start_date': '2014-09-17'  # Corresponds to period1=1410912000
    },
    'eth': {
        'symbol': 'ETH-USD',
        'name': 'Ethereum',
        'start_date': '2017-11-09'  # Corresponds to period1=1510185600
    }
}

def fetch_crypto_history(asset_code, symbol_info):
    """
    Fetch cryptocurrency price history using yfinance
    
    Args:
        asset_code: Asset code (e.g., 'btc', 'eth')
        symbol_info: Dictionary with symbol, name, and start_date
    
    Returns:
        DataFrame with date and price columns, or None if failed
    """
    symbol = symbol_info['symbol']
    start_date = symbol_info.get('start_date', '2010-01-01')
    
    try:
        print(f"  Fetching data for {asset_code.upper()} ({symbol})...")
        
        # Download historical data using yfinance
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=datetime.now().strftime('%Y-%m-%d'))
        
        if hist.empty:
            print(f"  ⚠️  No data returned for {asset_code.upper()}")
            return None
        
        # Process the data - yfinance returns a DataFrame with Close prices
        records = []
        for date, row in hist.iterrows():
            close_price = row['Close']
            if pd.notna(close_price):  # Skip NaN values
                # Normalize date to timezone-naive to ensure consistent storage
                date_naive = pd.to_datetime(date)
                if hasattr(date_naive, 'tz') and date_naive.tz is not None:
                    date_naive = date_naive.tz_localize(None)
                
                records.append({
                    'date': date_naive,
                    'price': float(close_price),
                    'product_id': None  # No product_id for crypto
                })
        
        if not records:
            print(f"  ⚠️  No valid price records found for {asset_code.upper()}")
            return None
        
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  ✓ Fetched {len(df)} records")
        print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error fetching data for {asset_code.upper()}: {e}")
        return None

def import_crypto_data(asset_code, symbol_info, db_path="db/investment_data.db"):
    """
    Import cryptocurrency data for a single asset
    
    Args:
        asset_code: Asset code (e.g., 'btc', 'eth')
        symbol_info: Dictionary with symbol, url, and name
        db_path: Path to SQLite database
    
    Returns:
        True if successful, False otherwise
    """
    asset_code_upper = asset_code.upper()
    asset_type = 'crypto'
    asset_name = symbol_info['name']
    
    print(f"\n{'='*60}")
    print(f"Importing {asset_code_upper} ({asset_name})")
    print(f"{'='*60}")
    
    # Fetch data from API
    df = fetch_crypto_history(asset_code, symbol_info)
    
    if df is None or len(df) == 0:
        print(f"✗ Failed to import {asset_code_upper}")
        return False
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        # OVERRIDE: Clear existing data for this asset
        print(f"  Clearing existing {asset_code_upper} data...")
        conn.execute('DELETE FROM price_data WHERE asset_code = ?', (asset_code_upper,))
        
        # Prepare data for database
        df_db = df.copy()
        df_db['asset_code'] = asset_code_upper
        df_db['asset_type'] = asset_type
        
        # Insert price data
        print(f"  Inserting {len(df)} price records...")
        df_db.to_sql('price_data', conn, if_exists='append', index=False)
        
        # Insert or update asset metadata
        inception_date = df['date'].min()
        last_date = df['date'].max()
        
        print(f"  Updating asset metadata...")
        conn.execute('''
            INSERT OR REPLACE INTO assets
            (asset_code, asset_name, asset_type, inception_date, last_update)
            VALUES (?, ?, ?, ?, ?)
        ''', (asset_code_upper, asset_name, asset_type, str(inception_date.date()), str(last_date.date())))
        
        conn.commit()
        
        print(f"✅ Successfully imported {asset_code_upper}:")
        print(f"   - Asset Type: {asset_type}")
        print(f"   - Asset Name: {asset_name}")
        print(f"   - Symbol: {symbol_info['symbol']}")
        print(f"   - Records: {len(df)}")
        print(f"   - Date Range: {inception_date.date()} to {last_date.date()}")
        
        # Verify import
        result = conn.execute('SELECT COUNT(*) FROM price_data WHERE asset_code = ?', (asset_code_upper,)).fetchone()
        print(f"   - Verified: {result[0]} records in database")
        
        return True
        
    except Exception as e:
        print(f"✗ Database error for {asset_code_upper}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def import_all_crypto(db_path="db/investment_data.db", delay_seconds=2):
    """
    Import all cryptocurrencies
    
    Args:
        db_path: Path to SQLite database
        delay_seconds: Delay between API calls to avoid rate limiting
    """
    print("="*60)
    print("Cryptocurrency Data Import")
    print("="*60)
    print(f"Cryptocurrencies: {len(CRYPTO_SYMBOLS)}")
    print(f"Database: {db_path}")
    print()
    
    success_count = 0
    fail_count = 0
    
    for i, (asset_code, symbol_info) in enumerate(CRYPTO_SYMBOLS.items(), 1):
        print(f"\n[Crypto {i}/{len(CRYPTO_SYMBOLS)}]")
        
        if import_crypto_data(asset_code, symbol_info, db_path):
            success_count += 1
        else:
            fail_count += 1
        
        # Add delay between requests to avoid rate limiting
        if i < len(CRYPTO_SYMBOLS):
            print(f"\n  Waiting {delay_seconds} seconds before next request...")
            time.sleep(delay_seconds)
    
    # Print summary
    print("\n" + "="*60)
    print("Import Summary")
    print("="*60)
    print(f"Total cryptocurrencies: {len(CRYPTO_SYMBOLS)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    # Show database status
    if success_count > 0:
        conn = sqlite3.connect(db_path)
        
        # Show crypto assets in database
        crypto_assets = pd.read_sql_query(
            """SELECT asset_code, asset_name, asset_type, inception_date, last_update 
               FROM assets 
               WHERE asset_type = 'crypto'
               ORDER BY asset_code""", 
            conn
        )
        if len(crypto_assets) > 0:
            print("\nCryptocurrencies in database:")
            print(crypto_assets.to_string(index=False))
            
            # Show latest prices
            latest_prices = pd.read_sql_query('''
                SELECT asset_code, MAX(date) as latest_date, price as latest_price
                FROM price_data
                WHERE asset_type = 'crypto'
                GROUP BY asset_code
                ORDER BY asset_code
            ''', conn)
            print(f"\nLatest price by cryptocurrency:")
            print(latest_prices.to_string(index=False))
        
        conn.close()

def main():
    """Main function"""
    import sys
    
    db_path = "db/investment_data.db"
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"Error: Database {db_path} not found")
        print("Please create the database first")
        return
    
    if len(sys.argv) > 1:
        # Single asset import mode
        asset_code = sys.argv[1].lower()
        
        if asset_code in CRYPTO_SYMBOLS:
            symbol_info = CRYPTO_SYMBOLS[asset_code]
            import_crypto_data(asset_code, symbol_info, db_path)
        else:
            print(f"Error: Unknown asset code '{asset_code}'")
            print(f"\nAvailable cryptocurrencies: {', '.join(sorted(CRYPTO_SYMBOLS.keys()))}")
            return
    else:
        # Import all cryptocurrencies
        import_all_crypto(db_path)

if __name__ == "__main__":
    main()
