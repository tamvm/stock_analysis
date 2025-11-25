#!/usr/bin/env python3

import pandas as pd
import json
import sqlite3
from datetime import datetime
from pathlib import Path

def import_stock_data(file_path, db_path="db/investment_data.db", asset_name=None):
    """
    Import stock/fund data from CSV or JSON file into price_data and assets tables

    Args:
        file_path: Path to the CSV or JSON data file
        db_path: Path to the SQLite database (default: "db/investment_data.db")
        asset_name: Optional custom asset name, if not provided will use asset_code

    Note: This function prevents duplicate imports by clearing existing data for the asset first.
    """

    # Ensure database directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Extract asset code from filename
    filename = Path(file_path).stem
    asset_code = filename.split("-")[0].upper()

    # Determine file format and read data
    print(f"Loading data from {file_path}...")
    print(f"Asset code detected: {asset_code}")

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

        else:
            # Read JSON file (original logic)
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
                    df_records.append({
                        'date': pd.to_datetime(record['navDate']),
                        'price': float(record['nav']),
                        'product_id': record.get('productId')
                    })

                    # Store product_id for metadata
                    if product_id is None and 'productId' in record:
                        product_id = record['productId']

        if not df_records:
            print("No valid price records found")
            return False

        # Create DataFrame from processed records
        df = pd.DataFrame(df_records)

    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    # Remove duplicate dates, keep the latest entry
    df = df.drop_duplicates(subset=['date'], keep='last')
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Processed {len(df)} unique price records")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Connect to database
    conn = sqlite3.connect(db_path)

    # Create tables if they don't exist
    conn.execute('''
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_code TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            date DATE NOT NULL,
            price REAL NOT NULL,
            product_id INTEGER,
            UNIQUE(asset_code, date)
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS assets (
            asset_code TEXT PRIMARY KEY,
            asset_name TEXT,
            asset_type TEXT,
            inception_date DATE,
            last_update DATE
        )
    ''')

    # Use provided asset_name or default to asset_code
    if asset_name is None:
        asset_name = asset_code

    asset_type = 'fund'

    try:
        # Clear existing data for this asset to prevent duplicates
        print(f"Clearing existing {asset_code} data...")
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

        print(f"✅ Successfully imported {asset_code} data:")
        print(f"   - Asset Code: {asset_code}")
        print(f"   - Asset Name: {asset_name}")
        print(f"   - Records: {len(df)}")
        print(f"   - Date Range: {inception_date.date()} to {last_date.date()}")
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

def main():
    """Main function to run the import"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python import_stock_data.py <file_path> [asset_name]")
        print("Example: python import_stock_data.py data/vti-112025.csv 'Vanguard Total Stock Market ETF'")
        print("Example: python import_stock_data.py data/qqq-112025.csv")
        print("Example: python import_stock_data.py data/dcbf-20251120.txt 'Dragon Capital Balanced Fund'")
        return

    file_path = sys.argv[1]
    asset_name = sys.argv[2] if len(sys.argv) > 2 else None
    db_path = "db/investment_data.db"

    # Extract asset code for display
    asset_code = Path(file_path).stem.split("-")[0].upper()
    print(f"=== {asset_code} Data Import ===")

    # Check if source file exists
    if not Path(file_path).exists():
        print(f"Error: Source file {file_path} not found")
        return

    # Run import
    success = import_stock_data(file_path, db_path, asset_name)

    if success:
        print("\n=== Import completed successfully! ===")

        # Show database status
        conn = sqlite3.connect(db_path)

        # Show all assets
        assets_df = pd.read_sql_query("SELECT * FROM assets ORDER BY asset_code", conn)
        print("\nAll assets in database:")
        print(assets_df.to_string(index=False))

        # Show price data sample for imported asset
        asset_code = Path(file_path).stem.split("-")[0].upper()
        sample_data = pd.read_sql_query('''
            SELECT asset_code, date, price
            FROM price_data
            WHERE asset_code = ?
            ORDER BY date
            LIMIT 5
        ''', conn, params=[asset_code])
        print(f"\n{asset_code} price data sample (first 5 records):")
        print(sample_data.to_string(index=False))

        # Show latest prices
        latest_prices = pd.read_sql_query('''
            SELECT asset_code, MAX(date) as latest_date,
                   price as latest_price
            FROM price_data
            GROUP BY asset_code
            ORDER BY asset_code
        ''', conn)
        print(f"\nLatest prices by asset:")
        print(latest_prices.to_string(index=False))

        conn.close()
    else:
        print("\n=== Import failed! ===")

if __name__ == "__main__":
    main()