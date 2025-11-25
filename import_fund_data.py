#!/usr/bin/env python3

"""
Generic script to import fund data into the investment_data database.
Handles empty files gracefully and can be re-run when data becomes available.
Supports importing any fund data file by providing the file path and asset code.

Usage:
    python import_fund_data.py data/vesaf-20251120.txt VESAF
    python import_fund_data.py data/vemeef-20251120.txt VEMEEF
    python import_fund_data.py --help
"""

import pandas as pd
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import argparse
import sys

class FundDataImporter:
    def __init__(self, db_path="db/investment_data.db"):
        self.db_path = db_path

    def setup_database(self):
        """Ensure database tables exist"""
        conn = sqlite3.connect(self.db_path)

        # Main price data table
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

        # Asset metadata table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS assets (
                asset_code TEXT PRIMARY KEY,
                asset_name TEXT,
                asset_type TEXT,
                inception_date DATE,
                last_update DATE,
                description TEXT,
                top_holdings TEXT,
                investment_strategy TEXT,
                sector_allocation TEXT,
                fund_size TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def load_fund_data(self, file_path, asset_code):
        """Load fund data from the JSON file"""
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None

        if file_path.stat().st_size == 0:
            print(f"⚠️  File is empty: {file_path}")
            print(f"This is normal if {asset_code} data hasn't been collected yet.")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'data' not in data or not data['data']:
                print(f"⚠️  No data found in file: {file_path}")
                return None

            df_records = []
            for record in data['data']:
                if 'nav' in record and 'navDate' in record:
                    # Parse the date and ensure it has proper timestamp format
                    nav_date = pd.to_datetime(record['navDate'])
                    # Ensure consistent timestamp format
                    if nav_date.time() == datetime.min.time():
                        nav_date = nav_date.replace(hour=0, minute=0, second=0)

                    df_records.append({
                        'date': nav_date,
                        'price': float(record['nav']),
                        'product_id': record.get('productId')
                    })

            if df_records:
                df = pd.DataFrame(df_records)
                # Remove duplicate dates, keep the latest entry
                df = df.drop_duplicates(subset=['date'], keep='last')
                df = df.sort_values('date').reset_index(drop=True)
                print(f"✅ Loaded {len(df)} records for {asset_code}")
                return df
            else:
                print(f"⚠️  No valid records found in {asset_code} data")
                return None

        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None

    def import_to_database(self, file_path, asset_code):
        """Import fund data to database"""
        self.setup_database()

        conn = sqlite3.connect(self.db_path)

        # Load fund data
        df = self.load_fund_data(file_path, asset_code)

        if df is None:
            # Still create asset entry even if no data
            print(f"Creating placeholder entry for {asset_code} in assets table...")
            conn.execute('''
                INSERT OR REPLACE INTO assets
                (asset_code, asset_name, asset_type, inception_date, last_update)
                VALUES (?, ?, ?, ?, ?)
            ''', (asset_code, asset_code, 'fund', None, datetime.now().strftime('%Y-%m-%d')))

            conn.commit()
            conn.close()
            print(f"✅ Created placeholder for {asset_code} (no data to import)")
            return 0

        # Clear existing data for this asset
        existing_count = conn.execute('SELECT COUNT(*) FROM price_data WHERE asset_code = ?', (asset_code,)).fetchone()[0]
        if existing_count > 0:
            conn.execute('DELETE FROM price_data WHERE asset_code = ?', (asset_code,))
            print(f"🗑️  Deleted {existing_count} existing records for {asset_code}")

        # Prepare data for database insertion with explicit timestamp formatting
        df_db = df.copy()
        df_db['asset_code'] = asset_code
        df_db['asset_type'] = 'fund'

        # Ensure date is formatted as timestamp string for consistency
        df_db['date'] = df_db['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Insert the data
        df_db.to_sql('price_data', conn, if_exists='append', index=False)
        print(f"✅ Inserted {len(df_db)} records for {asset_code}")

        # Update asset metadata
        inception_date = df['date'].min()
        last_date = df['date'].max()

        conn.execute('''
            INSERT OR REPLACE INTO assets
            (asset_code, asset_name, asset_type, inception_date, last_update)
            VALUES (?, ?, ?, ?, ?)
        ''', (asset_code, asset_code, 'fund', str(inception_date.date()), str(last_date.date())))

        print(f"📊 Updated metadata for {asset_code}")
        print(f"   Inception: {inception_date.date()}")
        print(f"   Last date: {last_date.date()}")

        conn.commit()
        conn.close()

        return len(df_db)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Import fund data into the investment database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python import_fund_data.py data/vesaf-20251120.txt VESAF
  python import_fund_data.py data/vemeef-20251120.txt VEMEEF
  python import_fund_data.py --list-assets
        '''
    )

    parser.add_argument('file_path', nargs='?', help='Path to the fund data file (JSON format)')
    parser.add_argument('asset_code', nargs='?', help='Asset code (e.g., VESAF, VEMEEF)')
    parser.add_argument('--list-assets', action='store_true', help='List all assets in the database')
    parser.add_argument('--db', default='db/investment_data.db', help='Database file path (default: db/investment_data.db)')

    return parser.parse_args()

def list_assets(db_path):
    """List all assets in the database"""
    try:
        conn = sqlite3.connect(db_path)
        assets = pd.read_sql_query(
            "SELECT asset_code, asset_type, inception_date, last_update FROM assets ORDER BY asset_type, asset_code",
            conn
        )
        conn.close()

        if len(assets) == 0:
            print("No assets found in the database.")
            return

        print(f"\n📋 Assets in database ({len(assets)} total):")
        print("-" * 60)

        current_type = None
        for _, asset in assets.iterrows():
            if asset['asset_type'] != current_type:
                current_type = asset['asset_type']
                print(f"\n{current_type.upper()}S:")

            inception = asset['inception_date'] or 'No data yet'
            last_update = asset['last_update'] or 'No data yet'
            print(f"  {asset['asset_code']} - Inception: {inception}, Last Update: {last_update}")

    except Exception as e:
        print(f"❌ Error listing assets: {e}")

def main():
    """Main function"""
    args = parse_arguments()

    # Handle list assets option
    if args.list_assets:
        list_assets(args.db)
        return

    # Validate required arguments
    if not args.file_path or not args.asset_code:
        print("❌ Error: Both file_path and asset_code are required")
        print("Use --help for usage information")
        sys.exit(1)

    # Validate file path
    if not Path(args.file_path).exists():
        print(f"❌ Error: File not found: {args.file_path}")
        sys.exit(1)

    # Validate asset code format
    asset_code = args.asset_code.upper()
    if not asset_code.replace('_', '').replace('-', '').isalnum():
        print(f"❌ Error: Invalid asset code format: {args.asset_code}")
        print("Asset codes should contain only letters, numbers, hyphens, and underscores")
        sys.exit(1)

    print(f"🚀 Starting {asset_code} data import from {args.file_path}...")

    importer = FundDataImporter(db_path=args.db)
    count = importer.import_to_database(args.file_path, asset_code)

    if count > 0:
        print(f"\n✅ Successfully imported {count} {asset_code} records with proper timestamp format")
    else:
        print(f"\n⚠️  No {asset_code} data to import (file empty or missing)")
        print("The script created a placeholder entry and can be re-run when data becomes available.")

    # Show updated asset list
    print(f"\n📋 Updated database status:")
    list_assets(args.db)

if __name__ == "__main__":
    main()