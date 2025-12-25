#!/usr/bin/env python3

"""
Metrics Job Manager
Handles background calculation of asset metrics using threading

Features:
- Asynchronous metric calculation
- Progress tracking
- Error handling per asset
- Database persistence
"""

import sqlite3
import pandas as pd
import threading
import time
from datetime import datetime
from typing import Callable, List, Dict, Optional
from calculate_metrics import MetricsCalculator
from config import DB_PATH

class MetricsJobManager:
    """Manages background metric calculation jobs"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.calculator = MetricsCalculator()
        self.current_job = None
        
    def get_all_assets(self):
        """Get list of all assets from database"""
        conn = sqlite3.connect(self.db_path)
        assets_df = pd.read_sql_query(
            "SELECT asset_code, asset_type FROM assets ORDER BY asset_code",
            conn
        )
        conn.close()
        return assets_df
        
    def get_asset_prices(self, asset_code):
        """Get price data for an asset"""
        conn = sqlite3.connect(self.db_path)
        prices_df = pd.read_sql_query(
            f"SELECT date, price FROM price_data WHERE asset_code = ? ORDER BY date",
            conn,
            params=(asset_code,)
        )
        conn.close()
        return prices_df
        
    def save_metrics(self, metrics):
        """Save calculated metrics to database"""
        conn = sqlite3.connect(self.db_path)
        
        # Delete existing metrics for this asset
        conn.execute("DELETE FROM asset_metrics WHERE asset_code = ?", (metrics['asset_code'],))
        
        # Insert new metrics
        columns = ', '.join(metrics.keys())
        placeholders = ', '.join(['?' for _ in metrics])
        sql = f"INSERT INTO asset_metrics ({columns}) VALUES ({placeholders})"
        
        conn.execute(sql, tuple(metrics.values()))
        conn.commit()
        conn.close()
        
    def create_calculation_run(self, assets_total):
        """Create a new calculation run record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "INSERT INTO calculation_runs (status, assets_total) VALUES (?, ?)",
            ('running', assets_total)
        )
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return run_id
        
    def update_calculation_run(self, run_id, status, assets_processed, error_message=None):
        """Update calculation run status"""
        conn = sqlite3.connect(self.db_path)
        
        if status == 'completed' or status == 'failed':
            conn.execute(
                """UPDATE calculation_runs 
                   SET status = ?, assets_processed = ?, completed_at = ?, error_message = ?
                   WHERE id = ?""",
                (status, assets_processed, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), error_message, run_id)
            )
        else:
            conn.execute(
                "UPDATE calculation_runs SET status = ?, assets_processed = ? WHERE id = ?",
                (status, assets_processed, run_id)
            )
            
        conn.commit()
        conn.close()
        
    def calculate_asset_metrics(self, asset_code, asset_type):
        """Calculate metrics for a single asset"""
        try:
            # Get price data
            prices_df = self.get_asset_prices(asset_code)
            
            if len(prices_df) == 0:
                return None, f"No price data found for {asset_code}"
                
            # Calculate metrics
            metrics = self.calculator.calculate_all_metrics(asset_code, prices_df, asset_type)
            
            # Save to database
            self.save_metrics(metrics)
            
            return metrics, None
            
        except Exception as e:
            return None, str(e)
            
    def calculate_all_metrics_background(self, progress_callback: Optional[Callable] = None):
        """
        Calculate metrics for all assets in background thread
        
        Args:
            progress_callback: Function to call with progress updates
                              Signature: callback(asset_code, processed, total, status, error)
        """
        # Get all assets
        assets_df = self.get_all_assets()
        total_assets = len(assets_df)
        
        if total_assets == 0:
            if progress_callback:
                progress_callback(None, 0, 0, 'completed', 'No assets found')
            return
            
        # Create calculation run
        run_id = self.create_calculation_run(total_assets)
        
        processed = 0
        errors = []
        
        # Process each asset
        for idx, asset in assets_df.iterrows():
            asset_code = asset['asset_code']
            asset_type = asset['asset_type']
            
            # Update progress
            if progress_callback:
                progress_callback(asset_code, processed, total_assets, 'running', None)
                
            # Calculate metrics
            metrics, error = self.calculate_asset_metrics(asset_code, asset_type)
            
            if error:
                errors.append(f"{asset_code}: {error}")
                
            processed += 1
            
            # Update calculation run
            self.update_calculation_run(run_id, 'running', processed)
            
        # Mark as completed
        final_status = 'completed' if len(errors) == 0 else 'completed'  # Still completed even with some errors
        error_message = '; '.join(errors) if errors else None
        self.update_calculation_run(run_id, final_status, processed, error_message)
        
        # Final progress update
        if progress_callback:
            progress_callback(None, processed, total_assets, 'completed', error_message)
            
    def start_calculation_job(self, progress_callback: Optional[Callable] = None):
        """
        Start background calculation job in a separate thread
        
        Args:
            progress_callback: Function to call with progress updates
            
        Returns:
            threading.Thread: The background thread
        """
        thread = threading.Thread(
            target=self.calculate_all_metrics_background,
            args=(progress_callback,),
            daemon=True
        )
        thread.start()
        self.current_job = thread
        return thread
        
    def get_last_calculation_run(self):
        """Get the most recent calculation run"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            """SELECT * FROM calculation_runs 
               WHERE status = 'completed'
               ORDER BY completed_at DESC 
               LIMIT 1""",
            conn
        )
        conn.close()
        
        if len(df) == 0:
            return None
            
        return df.iloc[0].to_dict()
        
    def get_asset_metrics(self, asset_code):
        """Get calculated metrics for an asset"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM asset_metrics WHERE asset_code = ?",
            conn,
            params=(asset_code,)
        )
        conn.close()
        
        if len(df) == 0:
            return None
            
        return df.iloc[0].to_dict()
        
    def get_all_metrics(self):
        """Get all calculated metrics"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM asset_metrics ORDER BY asset_code",
            conn
        )
        conn.close()
        return df

def main():
    """Test the job manager"""
    print("🧪 Testing Metrics Job Manager\n")
    
    manager = MetricsJobManager()
    
    # Get assets
    assets = manager.get_all_assets()
    print(f"Found {len(assets)} assets\n")
    
    # Define progress callback
    def progress_callback(asset_code, processed, total, status, error):
        if status == 'running':
            print(f"Processing {asset_code}... ({processed}/{total})")
        elif status == 'completed':
            print(f"\n✅ Completed! Processed {processed}/{total} assets")
            if error:
                print(f"⚠️  Errors: {error}")
                
    # Start calculation
    print("Starting background calculation...\n")
    thread = manager.start_calculation_job(progress_callback)
    
    # Wait for completion
    thread.join()
    
    # Show results
    print("\n📊 Metrics Summary:\n")
    metrics_df = manager.get_all_metrics()
    
    for _, row in metrics_df.iterrows():
        print(f"{row['asset_code']}:")
        if row['return_1y'] is not None:
            print(f"  1Y Return: {row['return_1y']:.2%}")
        if row['sharpe_ratio'] is not None:
            print(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}")
        print()
        
    # Show last run
    last_run = manager.get_last_calculation_run()
    if last_run:
        print(f"Last calculation: {last_run['completed_at']}")
        print(f"Assets processed: {last_run['assets_processed']}/{last_run['assets_total']}")

if __name__ == "__main__":
    main()
