#!/usr/bin/env python3

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path

def import_commodity_data(ticker, asset_code, asset_name, db_path="db/investment_data.db", 
                          start_date="2000-08-30", end_date=None):
    """
    Download commodity price data from Yahoo Finance and import into the database
    
    Args:
        ticker: Yahoo Finance ticker symbol (e.g., "SI=F" for silver)
        asset_code: Asset code to use in database (e.g., "SILVER")
        asset_name: Human-readable asset name (e.g., "Silver Futures")
        db_path: Path to the SQLite database (default: "db/investment_data.db")
        start_date: Start date for historical data (default: "2000-08-30")
        end_date: End date for historical data (default: today)
    
    Returns:
        True if successful, False otherwise
    """
    
    asset_type = "commodity"
    
    print("=" * 60)
    print(f"{asset_name} Price Data Import")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Asset Code: {asset_code}")
    print(f"Start Date: {start_date}")
    
    # Set end date to today if not provided
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    print(f"End Date: {end_date}")
    
    try:
        # Download commodity price data from Yahoo Finance
        print("\nDownloading data from Yahoo Finance...")
        commodity = yf.Ticker(ticker)
        df = commodity.history(start=start_date, end=end_date)
        
        if df.empty:
            print("Error: No data retrieved from Yahoo Finance")
            return False
        
        print(f"Downloaded {len(df)} records")
        
        # Process the data
        df_records = []
        for date, row in df.iterrows():
            # Normalize to timezone-naive
            if hasattr(date, 'tz') and date.tz is not None:
                date = date.tz_localize(None)
            
            df_records.append({
                'date': date,
                'price': float(row['Close']),
                'product_id': None
            })
        
        # Create DataFrame from processed records
        df_processed = pd.DataFrame(df_records)
        
        # Remove duplicate dates, keep the latest entry
        df_processed = df_processed.drop_duplicates(subset=['date'], keep='last')
        df_processed = df_processed.sort_values('date').reset_index(drop=True)
        
        print(f"Processed {len(df_processed)} unique price records")
        print(f"Date range: {df_processed['date'].min().date()} to {df_processed['date'].max().date()}")
        
    except Exception as e:
        print(f"Error downloading or processing data: {e}")
        return False
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        # OVERRIDE: Clear existing data for this asset
        print(f"\nOverriding existing {asset_code} data...")
        conn.execute('DELETE FROM price_data WHERE asset_code = ?', (asset_code,))
        
        # Prepare data for database
        df_db = df_processed.copy()
        df_db['asset_code'] = asset_code
        df_db['asset_type'] = asset_type
        
        # Insert price data
        print("Inserting price data...")
        df_db.to_sql('price_data', conn, if_exists='append', index=False)
        
        # Insert or update asset metadata
        inception_date = df_processed['date'].min()
        last_date = df_processed['date'].max()
        
        print("Updating asset metadata...")
        conn.execute('''
            INSERT OR REPLACE INTO assets
            (asset_code, asset_name, asset_type, inception_date, last_update)
            VALUES (?, ?, ?, ?, ?)
        ''', (asset_code, asset_name, asset_type, str(inception_date.date()), str(last_date.date())))
        
        conn.commit()
        
        print(f"\n✅ Successfully imported {asset_code}:")
        print(f"   - Asset Type: {asset_type}")
        print(f"   - Asset Name: {asset_name}")
        print(f"   - Records: {len(df_processed)}")
        print(f"   - Date Range: {inception_date.date()} to {last_date.date()}")
        
        # Verify import
        result = conn.execute('SELECT COUNT(*) FROM price_data WHERE asset_code = ?', (asset_code,)).fetchone()
        print(f"   - Verified: {result[0]} records in database")
        
        # Show sample of latest prices
        latest_prices = pd.read_sql_query('''
            SELECT date, price
            FROM price_data
            WHERE asset_code = ?
            ORDER BY date DESC
            LIMIT 5
        ''', conn, params=(asset_code,))
        
        print(f"\n   Latest prices:")
        for _, row in latest_prices.iterrows():
            print(f"   {row['date']}: ${row['price']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error importing data to database: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def main():
    """Main function to run the import"""
    import sys
    
    # Commodity configurations
    commodities = {
        'SILVER': {
            'ticker': 'SI=F',
            'asset_code': 'SILVER',
            'asset_name': 'Silver Futures'
        },
        'GOLD': {
            'ticker': 'GC=F',
            'asset_code': 'GOLD',
            'asset_name': 'Gold Futures'
        }
    }
    
    # Default parameters
    db_path = "db/investment_data.db"
    start_date = "2000-08-30"
    end_date = None
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        start_date = sys.argv[1]
    if len(sys.argv) > 2:
        end_date = sys.argv[2]
    
    # Import all commodities
    success_count = 0
    fail_count = 0
    
    for commodity_key, config in commodities.items():
        print()  # Add spacing between imports
        success = import_commodity_data(
            ticker=config['ticker'],
            asset_code=config['asset_code'],
            asset_name=config['asset_name'],
            db_path=db_path,
            start_date=start_date,
            end_date=end_date
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Import Summary")
    print("=" * 60)
    print(f"Total commodities: {len(commodities)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    if fail_count > 0:
        print("=" * 60)
        sys.exit(1)
    else:
        print("All imports completed successfully!")
        print("=" * 60)

if __name__ == "__main__":
    main()
