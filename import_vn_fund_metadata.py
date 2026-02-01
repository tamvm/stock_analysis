#!/usr/bin/env python3

import requests
import sqlite3
from datetime import datetime
from pathlib import Path
import time

# Vietnamese fund asset code to product ID mapping (fmarket API)
VN_FUND_PRODUCT_IDS = {
    'vesaf': 23,  # Vina Capital, equity
    'dcbf': 27,  # Dragon Capital, bond
    'dcde': 25,  # Dragon Capital, equity
    'dcds': 28,  # Dragon Capital, equity
    'magef': 35,  # Mirae Asset, equity
    'ssisca': 11,  # SSI Securities Corporation, equity
    'uveef': 58,  # UOBAM VIETNAM, equity
    'vcamdf': 75,  # Viet Capital Asset Management, equity
    'vcbfbcf': 32,  # Vietcombank, equity
    'vcbftbf': 31,  # Vietcombank, balanced
    'vemeef': 68,  # Vina Capital, equity
    'mafeqi': 72,  # Manulife Balanced Fund, equity
    'bvfed': 12,  # Bao Viet Fund Management, equity
    'kdef': 86,  # Kim Vietnam Fund Management, equity
    'bvpf': 14,  # Bao Viet Fund Management, equity
    'enf': 81,  # Eastspring Investments Vietnam Dynamic Fund, balanced
    'vcbfmgf': 46,  # Vietcombank, equity
    'vlgf': 49,  # SSI Securities Corporation, equity
    'mbvf': 47,  # MB Capital Value Fund, equity
    'veof': 20,  # Vina Capital, equity
    'vdef': 80  # Vina Capital, equity
}

def fetch_fund_metadata(product_id, asset_code):
    """
    Fetch fund metadata from fmarket API for a given product ID
    
    Args:
        product_id: Product ID from fmarket
        asset_code: Asset code (e.g., 'dcds')
    
    Returns:
        Dictionary with fund metadata, or None if failed
    """
    url = f'https://api.fmarket.vn/res/products/{product_id}'
    
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'vi',
        'authorization': 'Bearer eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0YW12bS5pdEBnbWFpbC5jb20iLCJhdWRpZW5jZSI6IldFQiIsImNyZWF0ZWQiOjE3Njk3ODMzNTc5MDcsInVzZXJ0eXBlIjoiSU5WRVNUT1IiLCJleHAiOjE3Njk3ODM5NTcsInN1YmZvYWNjb3VudCI6bnVsbH0.h8CUK8TNCy-ORSwM3x3JqEhKiWWYi1C471MKkeOYzHtRuQ5MWciYV7ar_Ta9STHw6U1Rr3UsI4Ggvsw8ldFxwQ',
        'f-language': 'vi',
        'origin': 'https://fmarket.vn',
        'referer': 'https://fmarket.vn/',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }
    
    try:
        print(f"  Fetching metadata for {asset_code.upper()} (product_id: {product_id})...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' not in data or not data['data']:
            print(f"  ⚠️  No data returned for {asset_code.upper()}")
            return None
        
        return data['data']
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ API request failed for {asset_code.upper()}: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error processing data for {asset_code.upper()}: {e}")
        return None

def import_fund_metadata(asset_code, product_id, db_path="db/investment_data.db"):
    """
    Import fund metadata for a single fund
    
    Args:
        asset_code: Asset code (e.g., 'dcds')
        product_id: Product ID from fmarket
        db_path: Path to SQLite database
    
    Returns:
        True if successful, False otherwise
    """
    asset_code_upper = asset_code.upper()
    
    print(f"\n{'='*60}")
    print(f"Importing metadata for {asset_code_upper}")
    print(f"{'='*60}")
    
    # Fetch metadata from API
    metadata = fetch_fund_metadata(product_id, asset_code)
    
    if metadata is None:
        print(f"✗ Failed to import metadata for {asset_code_upper}")
        return False
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Extract fund basic information
        fund_report = metadata.get('fundReport', {})
        owner = metadata.get('owner', {})
        
        # Parse report date
        report_date = None
        if fund_report.get('reportTime'):
            report_date = datetime.fromtimestamp(fund_report['reportTime'] / 1000).date()
        
        # Insert/update fund_info
        print(f"  Updating fund_info...")
        cursor.execute('''
            INSERT OR REPLACE INTO fund_info (
                product_id, asset_code, fund_name, fund_short_name, fund_code,
                issuer_name, issuer_short_name, fund_asset_type, fund_asset_code,
                description, total_assets_value, total_assets_value_str,
                total_assets_report_date, management_fee, performance_fee, nav,
                last_update
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            product_id,
            asset_code_upper,
            metadata.get('name'),
            metadata.get('shortName'),
            metadata.get('code'),
            owner.get('name'),
            owner.get('shortName'),
            metadata.get('fundAssetType'),
            metadata.get('dataFundAssetType', {}).get('code') if metadata.get('dataFundAssetType') else None,
            metadata.get('description'),
            fund_report.get('totalAssetValue'),
            fund_report.get('totalAssetValueStr'),
            str(report_date) if report_date else None,
            metadata.get('managementFee'),
            metadata.get('performanceFee'),
            metadata.get('nav')
        ))
        
        # Clear existing fees, holdings, and sector allocation for this fund
        print(f"  Clearing existing metadata...")
        cursor.execute('DELETE FROM fund_fees WHERE product_id = ?', (product_id,))
        cursor.execute('DELETE FROM fund_holdings WHERE product_id = ?', (product_id,))
        cursor.execute('DELETE FROM fund_sector_allocation WHERE product_id = ?', (product_id,))
        
        # Insert fees
        fee_list = metadata.get('productFeeList') or []
        if fee_list:
            print(f"  Inserting {len(fee_list)} fee records...")
            for fee in fee_list:
                cursor.execute('''
                    INSERT INTO fund_fees (
                        product_id, asset_code, fee_type, fee_unit_type, begin_volume,
                        end_volume, fee_percentage, last_update
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    product_id,
                    asset_code_upper,
                    fee.get('type'),
                    fee.get('feeUnitType'),
                    fee.get('beginVolume'),
                    fee.get('endVolume'),
                    fee.get('fee')
                ))
        
        # Insert top holdings (limit to top 10)
        holdings_list = metadata.get('productTopHoldingList') or []
        if holdings_list:
            holdings_list = holdings_list[:10]
            print(f"  Inserting {len(holdings_list)} top holdings...")
            for holding in holdings_list:
                cursor.execute('''
                    INSERT INTO fund_holdings (
                        product_id, asset_code, stock_code, industry, holding_type,
                        net_asset_percent, price, price_change, price_change_percent,
                        last_update
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    product_id,
                    asset_code_upper,
                    holding.get('stockCode'),
                    holding.get('industry'),
                    holding.get('type'),
                    holding.get('netAssetPercent'),
                    holding.get('price'),
                    holding.get('changeFromPrevious'),
                    holding.get('changeFromPreviousPercent')
                ))
        
        # Insert sector allocation
        sector_list = metadata.get('productIndustriesHoldingList') or []
        if sector_list:
            print(f"  Inserting {len(sector_list)} sector allocations...")
            for sector in sector_list:
                cursor.execute('''
                    INSERT INTO fund_sector_allocation (
                        product_id, asset_code, industry, asset_percent, last_update
                    ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    product_id,
                    asset_code_upper,
                    sector.get('industry'),
                    sector.get('assetPercent')
                ))
        
        conn.commit()
        
        print(f"✅ Successfully imported metadata for {asset_code_upper}:")
        print(f"   - Fund: {metadata.get('shortName')} ({metadata.get('name')})")
        print(f"   - Issuer: {owner.get('shortName')}")
        print(f"   - Asset Type: {metadata.get('fundAssetType')}")
        print(f"   - Total Assets: {fund_report.get('totalAssetValueStr', 'N/A')}")
        print(f"   - Fees: {len(fee_list)} records")
        print(f"   - Holdings: {len(holdings_list)} records")
        print(f"   - Sectors: {len(sector_list)} records")
        
        return True
        
    except Exception as e:
        print(f"✗ Database error for {asset_code_upper}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def import_all_fund_metadata(db_path="db/investment_data.db", delay_seconds=1):
    """
    Import metadata for all VN funds
    
    Args:
        db_path: Path to SQLite database
        delay_seconds: Delay between API calls to avoid rate limiting
    """
    print("="*60)
    print("Vietnamese Fund Metadata Import")
    print("="*60)
    print(f"Total funds: {len(VN_FUND_PRODUCT_IDS)}")
    print(f"Database: {db_path}")
    print()
    
    success_count = 0
    fail_count = 0
    
    for i, (asset_code, product_id) in enumerate(VN_FUND_PRODUCT_IDS.items(), 1):
        print(f"\n[Fund {i}/{len(VN_FUND_PRODUCT_IDS)}]")
        
        if import_fund_metadata(asset_code, product_id, db_path):
            success_count += 1
        else:
            fail_count += 1
        
        # Add delay between requests to avoid rate limiting
        if i < len(VN_FUND_PRODUCT_IDS):
            time.sleep(delay_seconds)
    
    # Print summary
    print("\n" + "="*60)
    print("Import Summary")
    print("="*60)
    print(f"Total funds: {len(VN_FUND_PRODUCT_IDS)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

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
        # Single fund import mode
        asset_code = sys.argv[1].lower()
        
        if asset_code in VN_FUND_PRODUCT_IDS:
            product_id = VN_FUND_PRODUCT_IDS[asset_code]
            import_fund_metadata(asset_code, product_id, db_path)
        else:
            print(f"Error: Unknown asset code '{asset_code}'")
            print(f"\nAvailable funds: {', '.join(sorted(VN_FUND_PRODUCT_IDS.keys()))}")
            return
    else:
        # Import all fund metadata
        import_all_fund_metadata(db_path)

if __name__ == "__main__":
    main()
