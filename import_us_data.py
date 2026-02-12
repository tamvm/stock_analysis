#!/usr/bin/env python3

import pandas as pd
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import glob

def extract_asset_code(file_path):
    """
    Extract asset code from filename
    
    Supports two formats:
    1. US format: HistoricalData_1767040077932-qqq.csv -> QQQ
    2. Standard format: dcbf-20251120.txt -> DCBF
    """
    filename = Path(file_path).stem
    
    # Check if it's US format (contains HistoricalData_)
    if filename.startswith('HistoricalData_'):
        parts = filename.split('-')
        if len(parts) >= 2:
            return parts[-1].upper()
    
    # Standard format - asset code before first dash
    parts = filename.split('-')
    if len(parts) >= 1:
        return parts[0].upper()
    
    return None

def import_stock_data(file_path, db_path="db/investment_data.db", asset_name=None, asset_type=None):
    """
    Import stock/fund data from CSV or JSON file into price_data and assets tables
    
    Args:
        file_path: Path to the CSV or JSON data file
        db_path: Path to the SQLite database (default: "db/investment_data.db")
        asset_name: Optional custom asset name, if not provided will use asset_code
        asset_type: Optional asset type (stock/fund/benchmark), auto-detected if not provided
    
    Returns:
        True if successful, False otherwise
    
    Note: This function overrides all existing data for the asset code.
    """
    
    # Extract asset code from filename
    asset_code = extract_asset_code(file_path)
    
    if not asset_code:
        print(f"Error: Cannot extract asset code from filename: {Path(file_path).name}")
        return False
    
    print(f"\nProcessing: {Path(file_path).name}")
    print(f"Asset code detected: {asset_code}")
    
    # Determine file format and read data
    file_extension = Path(file_path).suffix.lower()
    
    try:
        if file_extension == '.csv':
            # Read CSV file
            df = pd.read_csv(file_path)
            print(f"Found {len(df)} records in CSV file")
            
            # Check required columns
            if 'Date' not in df.columns or 'Close/Last' not in df.columns:
                print("Error: CSV file must have 'Date' and 'Close/Last' columns")
                return False
            
            # Process CSV data
            df_records = []
            for _, row in df.iterrows():
                try:
                    # Parse date and price
                    date = pd.to_datetime(row['Date'])
                    # Normalize to timezone-naive
                    if hasattr(date, 'tz') and date.tz is not None:
                        date = date.tz_localize(None)
                    
                    price_str = str(row['Close/Last']).replace('$', '').replace(',', '')
                    price = float(price_str)
                    
                    df_records.append({
                        'date': date,
                        'price': price,
                        'product_id': None
                    })
                except (ValueError, TypeError) as e:
                    print(f"Skipping invalid record: {row['Date']} - {e}")
                    continue
            
            product_id = None
            
            # Auto-detect asset type for CSV files (US stocks/ETFs)
            if asset_type is None:
                # List of known US ETFs
                us_etfs = ['QQQ', 'VTI', 'VOO']
                if asset_code in us_etfs:
                    asset_type = 'us_etf'
                else:
                    asset_type = 'us_stock'
        
        else:
            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract NAV data
            if 'data' not in data or not data['data']:
                print("No data found in JSON file")
                return False
            
            print(f"Found {len(data['data'])} records in file")
            
            # Process the data
            df_records = []
            product_id = None
            
            for record in data['data']:
                if 'nav' in record and 'navDate' in record:
                    date = pd.to_datetime(record['navDate'])
                    # Normalize to timezone-naive
                    if hasattr(date, 'tz') and date.tz is not None:
                        date = date.tz_localize(None)
                    
                    df_records.append({
                        'date': date,
                        'price': float(record['nav']),
                        'product_id': record.get('productId')
                    })
                    
                    # Store product_id for metadata
                    if product_id is None and 'productId' in record:
                        product_id = record['productId']
            
            # Auto-detect asset type for JSON files (likely funds)
            if asset_type is None:
                asset_type = 'fund'
        
        if not df_records:
            print("No valid price records found")
            return False
        
        # Create DataFrame from processed records
        df = pd.DataFrame(df_records)
        
        # Remove duplicate dates, keep the latest entry
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Processed {len(df)} unique price records")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Use provided asset_name or default to asset_code
    if asset_name is None:
        asset_name = asset_code
    
    try:
        # OVERRIDE: Clear existing data for this asset
        print(f"Overriding existing {asset_code} data...")
        conn.execute('DELETE FROM price_data WHERE asset_code = ?', (asset_code,))
        
        # Prepare data for database
        df_db = df.copy()
        df_db['asset_code'] = asset_code
        df_db['asset_type'] = asset_type
        
        # Insert price data
        print("Inserting price data...")
        df_db.to_sql('price_data', conn, if_exists='append', index=False)
        
        # Insert or update asset metadata
        inception_date = df['date'].min()
        last_date = df['date'].max()
        
        print("Updating asset metadata...")
        conn.execute('''
            INSERT OR REPLACE INTO assets
            (asset_code, asset_name, asset_type, inception_date, last_update)
            VALUES (?, ?, ?, ?, ?)
        ''', (asset_code, asset_name, asset_type, str(inception_date.date()), str(last_date.date())))
        
        conn.commit()
        
        print(f"✅ Successfully imported {asset_code}:")
        print(f"   - Asset Type: {asset_type}")
        print(f"   - Records: {len(df)}")
        print(f"   - Date Range: {inception_date.date()} to {last_date.date()}")
        if product_id:
            print(f"   - Product ID: {product_id}")
        
        # Verify import
        result = conn.execute('SELECT COUNT(*) FROM price_data WHERE asset_code = ?', (asset_code,)).fetchone()
        print(f"   - Verified: {result[0]} records in database")
        
        return True
        
    except Exception as e:
        print(f"Error importing data: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def download_and_import_us_funds(db_path="db/investment_data.db"):
    """
    Download historical data for US funds (FCNTX, FMAGX, BRK.B) from Yahoo Finance
    and import directly into the database.
    
    Args:
        db_path: Path to the SQLite database (default: "db/investment_data.db")
    
    Returns:
        True if all downloads and imports successful, False otherwise
    """
    import yfinance as yf
    
    # Define the funds to download
    funds = [
        {'ticker': 'FCNTX', 'name': 'Fidelity Contrafund', 'type': 'us_fund'},
        {'ticker': 'FMAGX', 'name': 'Fidelity Magellan', 'type': 'us_fund'},
        {'ticker': 'BRK-B', 'name': 'Berkshire Hathaway Class B', 'type': 'us_stock', 'code': 'BRK.B'}
    ]
    
    print("=" * 60)
    print("US Funds Download and Import")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for fund in funds:
        ticker = fund['ticker']
        asset_code = fund.get('code', ticker)  # Use 'code' if specified, otherwise use ticker
        asset_name = fund['name']
        asset_type = fund['type']
        
        print(f"\n{'=' * 60}")
        print(f"Processing: {asset_name} ({asset_code})")
        print(f"{'=' * 60}")
        
        try:
            # Download data from Yahoo Finance
            print(f"Downloading data from Yahoo Finance...")
            df = yf.download(ticker, period="max", progress=False)
            
            if df.empty:
                print(f"❌ No data returned for {ticker}")
                fail_count += 1
                continue
            
            print(f"Downloaded {len(df)} records")
            
            # Prepare data for database
            df_records = []
            for date, row in df.iterrows():
                # Normalize date to timezone-naive
                if hasattr(date, 'tz') and date.tz is not None:
                    date = date.tz_localize(None)
                
                df_records.append({
                    'date': date,
                    'price': float(row['Close'].iloc[0]) if hasattr(row['Close'], 'iloc') else float(row['Close']),
                    'asset_code': asset_code,
                    'asset_type': asset_type
                })
            
            # Create DataFrame
            df_db = pd.DataFrame(df_records)
            df_db = df_db.sort_values('date').reset_index(drop=True)
            
            print(f"Processed {len(df_db)} unique price records")
            print(f"Date range: {df_db['date'].min().date()} to {df_db['date'].max().date()}")
            
            # Connect to database and import
            conn = sqlite3.connect(db_path)
            
            try:
                # OVERRIDE: Clear existing data for this asset
                print(f"Overriding existing {asset_code} data...")
                conn.execute('DELETE FROM price_data WHERE asset_code = ?', (asset_code,))
                
                # Insert price data
                print("Inserting price data...")
                df_db.to_sql('price_data', conn, if_exists='append', index=False)
                
                # Insert or update asset metadata
                inception_date = df_db['date'].min()
                last_date = df_db['date'].max()
                
                print("Updating asset metadata...")
                conn.execute('''
                    INSERT OR REPLACE INTO assets
                    (asset_code, asset_name, asset_type, inception_date, last_update)
                    VALUES (?, ?, ?, ?, ?)
                ''', (asset_code, asset_name, asset_type, str(inception_date.date()), str(last_date.date())))
                
                conn.commit()
                
                print(f"✅ Successfully imported {asset_code}:")
                print(f"   - Asset Type: {asset_type}")
                print(f"   - Records: {len(df_db)}")
                print(f"   - Date Range: {inception_date.date()} to {last_date.date()}")
                
                # Verify import
                result = conn.execute('SELECT COUNT(*) FROM price_data WHERE asset_code = ?', (asset_code,)).fetchone()
                print(f"   - Verified: {result[0]} records in database")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error importing data: {e}")
                conn.rollback()
                fail_count += 1
            finally:
                conn.close()
                
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print("Download and Import Summary")
    print("=" * 60)
    print(f"Total funds: {len(funds)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    # Show database status if any succeeded
    if success_count > 0:
        conn = sqlite3.connect(db_path)
        
        # Show imported assets
        assets_df = pd.read_sql_query(
            "SELECT asset_code, asset_name, asset_type, inception_date, last_update FROM assets WHERE asset_code IN ('FCNTX', 'FMAGX', 'BRK.B') ORDER BY asset_type, asset_code", 
            conn
        )
        print("\nImported assets:")
        print(assets_df.to_string(index=False))
        
        # Show latest prices
        latest_prices = pd.read_sql_query('''
            SELECT asset_type, asset_code, MAX(date) as latest_date,
                   price as latest_price
            FROM price_data
            WHERE asset_code IN ('FCNTX', 'FMAGX', 'BRK.B')
            GROUP BY asset_code
            ORDER BY asset_type, asset_code
        ''', conn)
        print(f"\nLatest prices:")
        print(latest_prices.to_string(index=False))
        
        conn.close()
    
    return fail_count == 0

def batch_import_data(data_dirs=None, db_path="db/investment_data.db"):
    """
    Batch import all data files from specified directories
    
    Args:
        data_dirs: List of directories to import from. If None, defaults to ['data/us']
        db_path: Path to the SQLite database
    """
    
    if data_dirs is None:
        data_dirs = ['data/us']
    
    print("=" * 60)
    print("Stock/Fund Data Batch Import")
    print("=" * 60)
    
    all_files = []
    
    # Find all CSV and JSON files in specified directories
    for data_dir in data_dirs:
        if not Path(data_dir).exists():
            print(f"Warning: Directory {data_dir} does not exist, skipping...")
            continue
        
        csv_files = glob.glob(f"{data_dir}/*.csv")
        json_files = glob.glob(f"{data_dir}/*.txt")  # JSON files with .txt extension
        json_files.extend(glob.glob(f"{data_dir}/*.json"))
        
        all_files.extend(csv_files)
        all_files.extend(json_files)
        
        print(f"Found {len(csv_files)} CSV and {len(json_files)} JSON files in {data_dir}/")
    
    if not all_files:
        print("No data files found")
        return
    
    print(f"\nTotal files to import: {len(all_files)}")
    
    success_count = 0
    fail_count = 0
    
    for file_path in sorted(all_files):
        try:
            if import_stock_data(file_path, db_path):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print("Import Summary")
    print("=" * 60)
    print(f"Total files: {len(all_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    # Show database status
    if success_count > 0:
        conn = sqlite3.connect(db_path)
        
        # Show all assets by type
        assets_df = pd.read_sql_query(
            "SELECT asset_code, asset_name, asset_type, inception_date, last_update FROM assets ORDER BY asset_type, asset_code", 
            conn
        )
        print("\nAll assets in database:")
        print(assets_df.to_string(index=False))
        
        # Show latest prices by type
        latest_prices = pd.read_sql_query('''
            SELECT asset_type, asset_code, MAX(date) as latest_date,
                   price as latest_price
            FROM price_data
            GROUP BY asset_code
            ORDER BY asset_type, asset_code
        ''', conn)
        print(f"\nLatest prices by asset:")
        print(latest_prices.to_string(index=False))
        
        conn.close()

def main():
    """Main function to run the import"""
    import sys
    
    if len(sys.argv) > 1:
        # Single file import mode
        file_path = sys.argv[1]
        asset_name = sys.argv[2] if len(sys.argv) > 2 else None
        db_path = "db/investment_data.db"
        
        if not Path(file_path).exists():
            print(f"Error: File {file_path} not found")
            return
        
        import_stock_data(file_path, db_path, asset_name)
    else:
        # Batch import mode - import from data/us/ folder only
        print("Batch import mode: importing US stocks/ETFs from data/us/ folder")
        batch_import_data()

if __name__ == "__main__":
    main()
