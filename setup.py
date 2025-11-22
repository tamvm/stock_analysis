#!/usr/bin/env python3
"""
Setup script for Investment Analysis Portal
Run this script first to process all data files and create the database.
"""

import os
import sys
from data_processor import DataProcessor

def main():
    print("🚀 Setting up Investment Analysis Portal...")
    print("=" * 50)

    # Check if data directory exists
    if not os.path.exists("data"):
        print("❌ Error: 'data' directory not found!")
        print("Please ensure you have the data folder with fund files and references folder.")
        sys.exit(1)

    # Check if references directory exists
    if not os.path.exists("data/references"):
        print("❌ Error: 'data/references' directory not found!")
        print("Please ensure you have the benchmark files (vnindex-2017.csv, voo-2017.csv).")
        sys.exit(1)

    # Initialize data processor
    processor = DataProcessor()

    print("📊 Processing fund data and benchmark files...")

    try:
        # Process and save all data to database
        funds_count, benchmarks_count = processor.save_to_database()

        print(f"✅ Successfully processed:")
        print(f"   - {funds_count} mutual funds")
        print(f"   - {benchmarks_count} benchmark indices")

        # Show available assets
        print("\n📈 Available assets for analysis:")
        assets_df = processor.get_asset_list()

        print("\n🏦 Mutual Funds:")
        funds = assets_df[assets_df['asset_type'] == 'fund']
        for _, row in funds.iterrows():
            print(f"   - {row['asset_code']} (since {row['inception_date'][:10]})")

        print("\n📊 Benchmarks:")
        benchmarks = assets_df[assets_df['asset_type'] == 'benchmark']
        for _, row in benchmarks.iterrows():
            print(f"   - {row['asset_code']} (since {row['inception_date'][:10]})")

        print("\n" + "=" * 50)
        print("✅ Setup completed successfully!")
        print("\n🚀 To start the dashboard, run:")
        print("   streamlit run dashboard.py")
        print("\nThen open your browser and navigate to the provided URL (usually http://localhost:8501)")

    except Exception as e:
        print(f"❌ Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()