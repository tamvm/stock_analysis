import pandas as pd
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os

class DataProcessor:
    def __init__(self, db_path="db/investment_data.db"):
        self.db_path = db_path
        self.data_dir = Path("data")
        self.references_dir = Path("data/references")

    def setup_database(self):
        """Create SQLite database with unified schema"""
        conn = sqlite3.connect(self.db_path)

        # Main price data table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_code TEXT NOT NULL,
                asset_type TEXT NOT NULL,  -- 'fund' or 'benchmark'
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

    def load_fund_data(self):
        """Load all JSON fund data files"""
        fund_data = {}

        for file_path in self.data_dir.glob("*-20251120.txt"):
            if file_path.stat().st_size == 0:  # Skip empty files
                continue

            asset_code = file_path.stem.split("-")[0].upper()

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'data' in data and data['data']:
                    df_records = []
                    for record in data['data']:
                        if 'nav' in record and 'navDate' in record:
                            df_records.append({
                                'date': pd.to_datetime(record['navDate']),
                                'price': float(record['nav']),
                                'product_id': record.get('productId')
                            })

                    if df_records:
                        df = pd.DataFrame(df_records)
                        # Remove duplicate dates, keep the latest entry
                        df = df.drop_duplicates(subset=['date'], keep='last')
                        df = df.sort_values('date').reset_index(drop=True)
                        fund_data[asset_code] = df
                        print(f"Loaded {len(df)} records for {asset_code}")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return fund_data

    def load_benchmark_data(self):
        """Load benchmark CSV data"""
        benchmark_data = {}

        # VN-Index
        vnindex_path = self.references_dir / "vnindex-2017.csv"
        if vnindex_path.exists():
            df = pd.read_csv(vnindex_path)
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
            df = df[['Date', 'Price']].rename(columns={'Date': 'date', 'Price': 'price'})
            df = df.sort_values('date').reset_index(drop=True)
            benchmark_data['VNINDEX'] = df
            print(f"Loaded {len(df)} records for VN-Index")

        # S&P 500 (VOO)
        voo_path = self.references_dir / "voo-2017.csv"
        if voo_path.exists():
            df = pd.read_csv(voo_path)
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
            df = df[['Date', 'Price']].rename(columns={'Date': 'date', 'Price': 'price'})
            df = df.sort_values('date').reset_index(drop=True)
            benchmark_data['SP500'] = df
            print(f"Loaded {len(df)} records for S&P 500")

        return benchmark_data

    def save_to_database(self):
        """Load all data and save to SQLite database"""
        self.setup_database()

        conn = sqlite3.connect(self.db_path)

        # Load and save fund data
        fund_data = self.load_fund_data()
        for asset_code, df in fund_data.items():
            # Clear existing data for this asset
            conn.execute('DELETE FROM price_data WHERE asset_code = ?', (asset_code,))

            df_db = df.copy()
            df_db['asset_code'] = asset_code
            df_db['asset_type'] = 'fund'
            df_db.to_sql('price_data', conn, if_exists='append', index=False)

            # Save asset metadata
            inception_date = df['date'].min()
            last_date = df['date'].max()

            conn.execute('''
                INSERT OR REPLACE INTO assets
                (asset_code, asset_name, asset_type, inception_date, last_update)
                VALUES (?, ?, ?, ?, ?)
            ''', (asset_code, asset_code, 'fund', str(inception_date.date()), str(last_date.date())))

        # Load and save benchmark data
        benchmark_data = self.load_benchmark_data()
        for asset_code, df in benchmark_data.items():
            # Clear existing data for this asset
            conn.execute('DELETE FROM price_data WHERE asset_code = ?', (asset_code,))

            df_db = df.copy()
            df_db['asset_code'] = asset_code
            df_db['asset_type'] = 'benchmark'
            df_db['product_id'] = None
            df_db.to_sql('price_data', conn, if_exists='append', index=False)

            # Save asset metadata
            inception_date = df['date'].min()
            last_date = df['date'].max()

            conn.execute('''
                INSERT OR REPLACE INTO assets
                (asset_code, asset_name, asset_type, inception_date, last_update)
                VALUES (?, ?, ?, ?, ?)
            ''', (asset_code, asset_code, 'benchmark', str(inception_date.date()), str(last_date.date())))

        conn.commit()
        conn.close()

        print(f"Data saved to {self.db_path}")
        return len(fund_data), len(benchmark_data)

    def get_asset_list(self, include_benchmarks=True):
        """Get list of available assets"""
        conn = sqlite3.connect(self.db_path)

        if include_benchmarks:
            query = "SELECT asset_code, asset_type, inception_date FROM assets ORDER BY asset_type, asset_code"
        else:
            query = "SELECT asset_code, asset_type, inception_date FROM assets WHERE asset_type = 'fund' ORDER BY asset_code"

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def update_asset_info(self, asset_code, description=None, top_holdings=None,
                         investment_strategy=None, sector_allocation=None, fund_size=None):
        """Update asset information"""
        conn = sqlite3.connect(self.db_path)

        # Build the update query dynamically based on provided parameters
        update_fields = []
        params = []

        if description is not None:
            update_fields.append("description = ?")
            params.append(description)
        if top_holdings is not None:
            update_fields.append("top_holdings = ?")
            params.append(top_holdings)
        if investment_strategy is not None:
            update_fields.append("investment_strategy = ?")
            params.append(investment_strategy)
        if sector_allocation is not None:
            update_fields.append("sector_allocation = ?")
            params.append(sector_allocation)
        if fund_size is not None:
            update_fields.append("fund_size = ?")
            params.append(fund_size)

        if update_fields:
            query = f"UPDATE assets SET {', '.join(update_fields)} WHERE asset_code = ?"
            params.append(asset_code)
            conn.execute(query, params)
            conn.commit()

        conn.close()

    def get_asset_details(self, asset_code):
        """Get detailed information for a specific asset"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT asset_code, asset_name, asset_type, inception_date, description,
                   top_holdings, investment_strategy, sector_allocation, fund_size
            FROM assets
            WHERE asset_code = ?
        """
        df = pd.read_sql_query(query, conn, params=[asset_code])
        conn.close()

        if len(df) > 0:
            return df.iloc[0].to_dict()
        return None

    def get_price_data(self, asset_codes, start_date=None, end_date=None):
        """Get price data for specified assets"""
        conn = sqlite3.connect(self.db_path)

        placeholders = ','.join(['?' for _ in asset_codes])
        query = f'''
            SELECT asset_code, date, price
            FROM price_data
            WHERE asset_code IN ({placeholders})
        '''

        params = asset_codes[:]

        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)

        query += ' ORDER BY asset_code, date'

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Handle mixed date formats in the database
        try:
            df['date'] = pd.to_datetime(df['date'])
        except ValueError:
            # Try different approaches for mixed formats
            try:
                df['date'] = pd.to_datetime(df['date'], format='mixed')
            except (ValueError, TypeError):
                # Fallback approach - convert each date individually
                dates = []
                for date_str in df['date']:
                    try:
                        if ' ' in str(date_str):
                            # Has timestamp
                            dates.append(pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S'))
                        else:
                            # Just date
                            dates.append(pd.to_datetime(date_str, format='%Y-%m-%d'))
                    except:
                        # Last resort - let pandas infer
                        dates.append(pd.to_datetime(date_str, errors='coerce'))
                df['date'] = dates
        return df

if __name__ == "__main__":
    processor = DataProcessor()
    funds_count, benchmarks_count = processor.save_to_database()
    print(f"Processed {funds_count} funds and {benchmarks_count} benchmarks")

    # Show available assets
    assets = processor.get_asset_list()
    print("\nAvailable assets:")
    print(assets)