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
import asyncio
import json

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
    'us': {
        'name': 'US Market',
        'page_url': 'https://simplywall.st/markets/us',
        'graphql_url': 'https://simplywall.st/graphql',
        'method': 'playwright'  # Requires Playwright due to Cloudflare protection
    }
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

async def fetch_us_market_statistics_async():
    """
    Fetch US market statistics from SimplyWall.St using Playwright
    Uses browser automation to bypass Cloudflare and intercept GraphQL responses
    
    Returns:
        DataFrame with columns: date, pe_ratio, total_market_cap, total_earnings, total_revenue
        or None if failed
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print(f"  ✗ Playwright not installed. Install with: pip install playwright && playwright install chromium")
        return None
    
    config = MARKET_CONFIGS['us']
    graphql_data = []
    
    print(f"  Launching browser to fetch data from SimplyWall.St...")
    print(f"  ⏳ This may take 20-30 seconds due to Cloudflare protection...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='en-US'
        )
        
        # Remove automation indicators
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        page = await context.new_page()
        
        async def handle_response(response):
            """Intercept GraphQL responses containing IndustryTimeseries data"""
            if '/graphql' in response.url and response.status == 200:
                try:
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        data = await response.json()
                        
                        # Check if this contains IndustryTimeseries data
                        if 'data' in data and 'IndustryTimeseries' in data.get('data', {}):
                            print(f"  ✓ Intercepted PE timeseries data from GraphQL")
                            graphql_data.append(data)
                except:
                    pass
        
        page.on('response', handle_response)
        
        try:
            # Navigate to the page
            await page.goto(config['page_url'], timeout=60000)
            
            # Wait for Cloudflare challenge to complete
            max_wait = 30
            waited = 0
            while waited < max_wait:
                title = await page.title()
                if 'Just a moment' not in title and 'security' not in title.lower():
                    print(f"  ✓ Cloudflare challenge bypassed")
                    break
                await page.wait_for_timeout(1000)
                waited += 1
            
            # Wait for page to load data
            await page.wait_for_timeout(8000)
            
            # If no data intercepted yet, try scrolling and interacting with the page
            if not graphql_data:
                print(f"  ⏳ Attempting to trigger data loading...")
                
                # Scroll to ensure all content is loaded
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(2000)
                await page.evaluate("window.scrollTo(0, 0)")
                await page.wait_for_timeout(2000)
                
                # Try to find and click on timeframe buttons
                try:
                    # Look for 5Y, 3Y, or other timeframe buttons
                    timeframe_buttons = await page.query_selector_all('button')
                    for button in timeframe_buttons:
                        text = await button.inner_text()
                        if text.strip() in ['5Y', '10Y', '3Y', 'MAX']:
                            print(f"  ✓ Clicking {text} button...")
                            await button.click()
                            await page.wait_for_timeout(3000)
                            if graphql_data:
                                break
                except Exception as e:
                    print(f"  ⚠️  Error clicking buttons: {e}")
                
                # Wait a bit more
                await page.wait_for_timeout(5000)
            
        finally:
            await browser.close()
    
    # Process the intercepted data
    if not graphql_data:
        print(f"  ✗ Failed to intercept or fetch PE data")
        return None
    
    # Extract timeseries data
    ts_data = graphql_data[0]['data']['IndustryTimeseries']
    
    pe_array = ts_data.get('pe', [])
    market_cap_array = ts_data.get('marketCap', [])
    earnings_array = ts_data.get('earnings', [])
    revenue_array = ts_data.get('revenue', [])
    
    if not pe_array:
        print(f"  ✗ No PE data found in response")
        return None
    
    print(f"  ✓ Extracted {len(pe_array)} data points")
    
    # Process data - each array contains [timestamp_ms, value] pairs
    records = []
    for i in range(len(pe_array)):
        if pe_array[i] and len(pe_array[i]) >= 2:
            timestamp_ms = pe_array[i][0]
            pe_value = pe_array[i][1]
            
            # Get corresponding values from other arrays
            market_cap = market_cap_array[i][1] if i < len(market_cap_array) and market_cap_array[i] else None
            earnings = earnings_array[i][1] if i < len(earnings_array) and earnings_array[i] else None
            revenue = revenue_array[i][1] if i < len(revenue_array) and revenue_array[i] else None
            
            records.append({
                'date': pd.to_datetime(timestamp_ms, unit='ms'),
                'pe_ratio': float(pe_value) if pe_value else None,
                'total_market_cap': float(market_cap) if market_cap else None,
                'total_earnings': float(earnings) if earnings else None,
                'total_revenue': float(revenue) if revenue else None
            })
    
    if not records:
        print(f"  ⚠️  No valid records found")
        return None
    
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=['date'], keep='last')
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"  ✓ Processed {len(df)} unique records")
    print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    if df['pe_ratio'].notna().any():
        print(f"    PE range: {df['pe_ratio'].min():.2f} to {df['pe_ratio'].max():.2f}")
    
    return df

def fetch_us_market_statistics_from_json(json_path="data/us/simplywall_pe.json"):
    """
    Load US market statistics from a local JSON file
    
    Args:
        json_path: Path to the JSON file containing SimplyWall.St data
    
    Returns:
        DataFrame with columns: date, pe_ratio, total_market_cap, total_earnings, total_revenue
        or None if failed
    """
    try:
        print(f"  Loading data from {json_path}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract the timeseries data
        ts_data = data.get('data', {}).get('IndustryTimeseries', {})
        
        if not ts_data:
            print(f"  ✗ Invalid JSON structure")
            return None
        
        # Get the data dictionaries
        absolute_pe = ts_data.get('absolutePE', {})
        market_cap = ts_data.get('marketCap', {})
        earnings = ts_data.get('earnings', {})
        revenue = ts_data.get('revenue', {})
        
        if not absolute_pe:
            print(f"  ✗ No absolutePE data found in JSON")
            return None
        
        print(f"  ✓ Found {len(absolute_pe)} data points")
        
        # Process data - convert date strings to datetime and create records
        records = []
        for date_str, pe_value in absolute_pe.items():
            try:
                date = pd.to_datetime(date_str)
                
                # Get corresponding values from other dictionaries
                mc_value = market_cap.get(date_str)
                earnings_value = earnings.get(date_str)
                revenue_value = revenue.get(date_str)
                
                records.append({
                    'date': date,
                    'pe_ratio': float(pe_value) if pe_value else None,
                    'total_market_cap': float(mc_value) if mc_value else None,
                    'total_earnings': float(earnings_value) if earnings_value else None,
                    'total_revenue': float(revenue_value) if revenue_value else None
                })
            except Exception as e:
                print(f"  ⚠️  Error processing date {date_str}: {e}")
                continue
        
        if not records:
            print(f"  ⚠️  No valid records found")
            return None
        
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  ✓ Processed {len(df)} unique records")
        print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        if df['pe_ratio'].notna().any():
            print(f"    PE range: {df['pe_ratio'].min():.2f} to {df['pe_ratio'].max():.2f}")
        
        return df
        
    except FileNotFoundError:
        print(f"  ✗ File not found: {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"  ✗ Invalid JSON format: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return None

def fetch_us_market_statistics_from_csv(csv_path="data/us/sp-500-pe-ratio-price-to-earnings-chart.csv"):
    """
    Load US market PE statistics from Macrotrends CSV file
    
    Args:
        csv_path: Path to the CSV file containing historical S&P 500 PE data
    
    Returns:
        DataFrame with columns: date, pe_ratio
        or None if failed
    """
    try:
        print(f"  Loading historical PE data from {csv_path}...")
        
        # Read CSV file, skipping the header rows
        df = pd.read_csv(csv_path, skiprows=16, names=['date', 'pe_ratio'])
        
        # Remove any empty rows
        df = df.dropna()
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert PE ratio to float
        df['pe_ratio'] = pd.to_numeric(df['pe_ratio'], errors='coerce')
        
        # Remove rows with invalid PE values
        df = df.dropna(subset=['pe_ratio'])
        
        # Add placeholder columns for consistency (will be NULL in database)
        df['total_market_cap'] = None
        df['total_earnings'] = None
        df['total_revenue'] = None
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  ✓ Loaded {len(df)} records from CSV")
        print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"    PE range: {df['pe_ratio'].min():.2f} to {df['pe_ratio'].max():.2f}")
        
        return df
        
    except FileNotFoundError:
        print(f"  ✗ File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"  ✗ Error loading CSV: {e}")
        return None

def fetch_us_market_statistics_combined():
    """
    Load US market statistics from both CSV (historical) and JSON (recent) sources
    Merges data to provide complete historical coverage
    
    Returns:
        DataFrame with columns: date, pe_ratio, total_market_cap, total_earnings, total_revenue
        or None if failed
    """
    print(f"  Loading US market PE data from multiple sources...")
    
    # Load CSV data (historical, 1927-present)
    csv_df = fetch_us_market_statistics_from_csv()
    
    # Load JSON data (recent, 2016-present with additional metrics)
    json_df = fetch_us_market_statistics_from_json()
    
    if csv_df is None and json_df is None:
        print(f"  ✗ Failed to load data from both sources")
        return None
    
    # If only one source is available, use it
    if csv_df is None:
        print(f"  ✓ Using JSON data only")
        return json_df
    
    if json_df is None:
        print(f"  ✓ Using CSV data only")
        return csv_df
    
    # Both sources available - merge them
    print(f"  Merging CSV and JSON data...")
    
    # Use JSON data for dates >= 2016-02-01 (has more detailed metrics)
    # Use CSV data for dates < 2016-02-01 (historical PE only)
    cutoff_date = pd.to_datetime('2016-02-01')
    
    csv_historical = csv_df[csv_df['date'] < cutoff_date].copy()
    json_recent = json_df[json_df['date'] >= cutoff_date].copy()
    
    # Combine the dataframes
    combined_df = pd.concat([csv_historical, json_recent], ignore_index=True)
    
    # Sort by date and remove duplicates (prefer JSON data if overlap)
    combined_df = combined_df.sort_values('date')
    combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
    combined_df = combined_df.reset_index(drop=True)
    
    print(f"  ✓ Combined dataset created:")
    print(f"    Total records: {len(combined_df)}")
    print(f"    Historical (CSV): {len(csv_historical)} records ({csv_historical['date'].min().date()} to {csv_historical['date'].max().date()})")
    print(f"    Recent (JSON): {len(json_recent)} records ({json_recent['date'].min().date()} to {json_recent['date'].max().date()})")
    print(f"    Full date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
    print(f"    PE range: {combined_df['pe_ratio'].min():.2f} to {combined_df['pe_ratio'].max():.2f}")
    
    return combined_df

def fetch_us_market_statistics():
    """
    Synchronous wrapper for fetch_us_market_statistics_async
    """
    return asyncio.run(fetch_us_market_statistics_async())

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
    elif market_code == 'us':
        # Load from combined sources (CSV for historical + JSON for recent)
        df = fetch_us_market_statistics_combined()
        if df is None:
            print(f"  ⚠️  Combined loader failed, trying Playwright scraper as last resort...")
            df = fetch_us_market_statistics()
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
