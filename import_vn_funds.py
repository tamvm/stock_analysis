#!/usr/bin/env python3

import requests
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
import time

# Vietnamese fund asset code to product ID mapping (fmarket API)
VN_FUND_PRODUCT_IDS = {
    'vesaf': 23,
    'dcbf': 27,
    'dcde': 25,
    'dcds': 28,
    'magef': 35,
    'ssisca': 11,
    'uveef': 58,
    'vcamdf': 75,
    'vcbfbcf': 32,
    'vcbftbf': 31,
    'vemeef': 68,
    'mafeqi': 72,
    'bvfed': 12,
    'kdef': 86,
    'bvpf': 14,
    'enf': 81,
    'vcbfmgf': 46,
    'vlgf': 49,
    'mbvf': 47,
    'veof': 20,
    'vdef': 80
}

# Vietnamese index tickers (Simplize API)
VN_INDEX_TICKERS = {
    'vnindex': 'VNINDEX',
    'vn30': 'VN30'
}

def fetch_index_history(ticker, asset_code):
    """
    Fetch index price history from Simplize API for VNINDEX or VN30
    
    Args:
        ticker: Ticker symbol (e.g., 'VNINDEX', 'VN30')
        asset_code: Asset code (e.g., 'vnindex', 'vn30')
    
    Returns:
        DataFrame with date and price columns, or None if failed
    """
    # Use OHLCV API with daily interval for more granular data
    url = f'https://api2.simplize.vn/api/historical/prices/ohlcv?ticker={ticker}&size=100000&interval=1d&type=index'
    
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en-US,en;q=0.9',
        'origin': 'https://simplize.vn',
        'referer': 'https://simplize.vn/',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }
    
    try:
        print(f"  Fetching daily data for {asset_code.upper()} (ticker: {ticker})...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        response_data = response.json()
        
        # Simplize OHLCV API returns: {status: 200, message: "Success", data: [...]}
        if not response_data or 'data' not in response_data:
            print(f"  ⚠️  No data returned for {asset_code.upper()}")
            return None
        
        data = response_data['data']
        
        if not data or not isinstance(data, list):
            print(f"  ⚠️  Invalid data format for {asset_code.upper()}")
            return None
        
        # Process the data - OHLCV format is [timestamp, open, high, low, close, volume]
        records = []
        for record in data:
            if isinstance(record, list) and len(record) >= 5:
                timestamp = record[0]  # Unix timestamp in seconds
                close_price = record[4]  # close price
                
                records.append({
                    'date': pd.to_datetime(timestamp, unit='s'),
                    'price': float(close_price),
                    'product_id': None  # No product_id for indices
                })
        
        if not records:
            print(f"  ⚠️  No valid price records found for {asset_code.upper()}")
            return None
        
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  ✓ Fetched {len(df)} daily records")
        print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ API request failed for {asset_code.upper()}: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error processing data for {asset_code.upper()}: {e}")
        return None

def fetch_nav_history(product_id, asset_code):
    """
    Fetch NAV history from fmarket API for a given product ID
    
    Args:
        product_id: Product ID from fmarket
        asset_code: Asset code (e.g., 'vesaf')
    
    Returns:
        DataFrame with date and price columns, or None if failed
    """
    url = 'https://api.fmarket.vn/res/product/get-nav-history'
    
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'vi',
        'authorization': 'Bearer eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0YW12bS5pdEBnbWFpbC5jb20iLCJhdWRpZW5jZSI6IldFQiIsImNyZWF0ZWQiOjE3NjcwNDAzMjkxNjQsInVzZXJ0eXBlIjoiSU5WRVNUT1IiLCJleHAiOjE3NjcwNDA5MjksInN1YmZvYWNjb3VudCI6bnVsbH0.QVonwI-8Lxl2w3WwgLK3I1a1I1bFb0PGAxarOOj-ndl88UDiBD2UbWH8savEvaYcqUVw0ueL79ldf4svKfXVdg',
        'content-type': 'application/json',
        'f-language': 'en',
        'origin': 'https://fmarket.vn',
        'referer': 'https://fmarket.vn/',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }
    
    payload = {
        "isAllData": 1,
        "productId": product_id,
        "navPeriod": "navToBeginning"
    }
    
    try:
        print(f"  Fetching data for {asset_code.upper()} (product_id: {product_id})...")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' not in data or not data['data']:
            print(f"  ⚠️  No data returned for {asset_code.upper()}")
            return None
        
        # Process the data
        records = []
        for record in data['data']:
            if 'nav' in record and 'navDate' in record:
                # Try to parse date - could be string or timestamp
                nav_date = record['navDate']
                if isinstance(nav_date, (int, float)):
                    date = pd.to_datetime(nav_date, unit='ms')
                else:
                    date = pd.to_datetime(nav_date)
                
                # Normalize to timezone-naive
                if hasattr(date, 'tz') and date.tz is not None:
                    date = date.tz_localize(None)
                
                records.append({
                    'date': date,
                    'price': float(record['nav']),
                    'product_id': product_id
                })
        
        if not records:
            print(f"  ⚠️  No valid NAV records found for {asset_code.upper()}")
            return None
        
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  ✓ Fetched {len(df)} records")
        print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ API request failed for {asset_code.upper()}: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error processing data for {asset_code.upper()}: {e}")
        return None

def import_vn_index_data(asset_code, ticker, db_path="db/investment_data.db"):
    """
    Import VN index data for a single index (VNINDEX or VN30)
    
    Args:
        asset_code: Asset code (e.g., 'vnindex', 'vn30')
        ticker: Ticker symbol (e.g., 'VNINDEX', 'VN30')
        db_path: Path to SQLite database
    
    Returns:
        True if successful, False otherwise
    """
    asset_code_upper = asset_code.upper()
    asset_type = 'vn_index'
    
    print(f"\n{'='*60}")
    print(f"Importing {asset_code_upper}")
    print(f"{'='*60}")
    
    # Fetch data from API
    df = fetch_index_history(ticker, asset_code)
    
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
        ''', (asset_code_upper, asset_code_upper, asset_type, str(inception_date.date()), str(last_date.date())))
        
        conn.commit()
        
        print(f"✅ Successfully imported {asset_code_upper}:")
        print(f"   - Asset Type: {asset_type}")
        print(f"   - Ticker: {ticker}")
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

def import_vn_fund_data(asset_code, product_id, db_path="db/investment_data.db"):
    """
    Import VN fund data for a single asset
    
    Args:
        asset_code: Asset code (e.g., 'vesaf')
        product_id: Product ID from fmarket
        db_path: Path to SQLite database
    
    Returns:
        True if successful, False otherwise
    """
    asset_code_upper = asset_code.upper()
    asset_type = 'vn_fund'
    
    print(f"\n{'='*60}")
    print(f"Importing {asset_code_upper}")
    print(f"{'='*60}")
    
    # Fetch data from API
    df = fetch_nav_history(product_id, asset_code)
    
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
        ''', (asset_code_upper, asset_code_upper, asset_type, str(inception_date.date()), str(last_date.date())))
        
        conn.commit()
        
        print(f"✅ Successfully imported {asset_code_upper}:")
        print(f"   - Asset Type: {asset_type}")
        print(f"   - Product ID: {product_id}")
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

def import_all_vn_assets(db_path="db/investment_data.db", delay_seconds=1):
    """
    Import all VN funds and indices
    
    Args:
        db_path: Path to SQLite database
        delay_seconds: Delay between API calls to avoid rate limiting
    """
    print("="*60)
    print("Vietnamese Assets Data Import")
    print("="*60)
    print(f"VN Funds: {len(VN_FUND_PRODUCT_IDS)}")
    print(f"VN Indices: {len(VN_INDEX_TICKERS)}")
    print(f"Total assets: {len(VN_FUND_PRODUCT_IDS) + len(VN_INDEX_TICKERS)}")
    print(f"Database: {db_path}")
    print()
    
    success_count = 0
    fail_count = 0
    total_count = 0
    
    # Import VN Funds
    print("\n" + "="*60)
    print("IMPORTING VN FUNDS")
    print("="*60)
    for i, (asset_code, product_id) in enumerate(VN_FUND_PRODUCT_IDS.items(), 1):
        total_count += 1
        print(f"\n[Fund {i}/{len(VN_FUND_PRODUCT_IDS)}]")
        
        if import_vn_fund_data(asset_code, product_id, db_path):
            success_count += 1
        else:
            fail_count += 1
        
        # Add delay between requests to avoid rate limiting
        if i < len(VN_FUND_PRODUCT_IDS):
            time.sleep(delay_seconds)
    
    # Import VN Indices
    print("\n" + "="*60)
    print("IMPORTING VN INDICES")
    print("="*60)
    for i, (asset_code, ticker) in enumerate(VN_INDEX_TICKERS.items(), 1):
        total_count += 1
        print(f"\n[Index {i}/{len(VN_INDEX_TICKERS)}]")
        
        if import_vn_index_data(asset_code, ticker, db_path):
            success_count += 1
        else:
            fail_count += 1
        
        # Add delay between requests to avoid rate limiting
        if i < len(VN_INDEX_TICKERS):
            time.sleep(delay_seconds)
    
    # Print summary
    print("\n" + "="*60)
    print("Import Summary")
    print("="*60)
    print(f"Total assets: {len(VN_FUND_PRODUCT_IDS) + len(VN_INDEX_TICKERS)}")
    print(f"  - VN Funds: {len(VN_FUND_PRODUCT_IDS)}")
    print(f"  - VN Indices: {len(VN_INDEX_TICKERS)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    # Show database status
    if success_count > 0:
        conn = sqlite3.connect(db_path)
        
        # Show VN funds in database
        vn_funds = pd.read_sql_query(
            """SELECT asset_code, asset_name, asset_type, inception_date, last_update 
               FROM assets 
               WHERE asset_type = 'vn_fund'
               ORDER BY asset_code""", 
            conn
        )
        if len(vn_funds) > 0:
            print("\nVN Funds in database:")
            print(vn_funds.to_string(index=False))
            
            # Show latest NAV
            latest_nav = pd.read_sql_query('''
                SELECT asset_code, MAX(date) as latest_date, price as latest_nav
                FROM price_data
                WHERE asset_type = 'vn_fund'
                GROUP BY asset_code
                ORDER BY asset_code
            ''', conn)
            print(f"\nLatest NAV by fund:")
            print(latest_nav.to_string(index=False))
        
        # Show VN indices in database
        vn_indices = pd.read_sql_query(
            """SELECT asset_code, asset_name, asset_type, inception_date, last_update 
               FROM assets 
               WHERE asset_type = 'vn_index'
               ORDER BY asset_code""", 
            conn
        )
        if len(vn_indices) > 0:
            print("\nVN Indices in database:")
            print(vn_indices.to_string(index=False))
            
            # Show latest prices
            latest_prices = pd.read_sql_query('''
                SELECT asset_code, MAX(date) as latest_date, price as latest_price
                FROM price_data
                WHERE asset_type = 'vn_index'
                GROUP BY asset_code
                ORDER BY asset_code
            ''', conn)
            print(f"\nLatest price by index:")
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
        
        # Check if it's a fund
        if asset_code in VN_FUND_PRODUCT_IDS:
            product_id = VN_FUND_PRODUCT_IDS[asset_code]
            import_vn_fund_data(asset_code, product_id, db_path)
        # Check if it's an index
        elif asset_code in VN_INDEX_TICKERS:
            ticker = VN_INDEX_TICKERS[asset_code]
            import_vn_index_data(asset_code, ticker, db_path)
        else:
            print(f"Error: Unknown asset code '{asset_code}'")
            print(f"\nAvailable VN Funds: {', '.join(sorted(VN_FUND_PRODUCT_IDS.keys()))}")
            print(f"Available VN Indices: {', '.join(sorted(VN_INDEX_TICKERS.keys()))}")
            return
    else:
        # Import all VN assets (funds + indices)
        import_all_vn_assets(db_path)

if __name__ == "__main__":
    main()
