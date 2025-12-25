#!/usr/bin/env python3

"""
Asset Metrics Calculator
Calculates financial performance metrics for investment assets

Metrics calculated:
- 1-year return
- 3-year annualized return
- 5-year annualized return
- Annualized standard deviation (volatility)
- Sharpe ratio
- Maximum drawdown (3-year and 5-year)
- Rolling CAGR (1Y, 3Y, 4Y, 5Y windows)
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from config import RISK_FREE_RATE, get_trading_days, ROLLING_WINDOWS

class MetricsCalculator:
    """Calculate financial metrics for assets"""
    
    def __init__(self, risk_free_rate=RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        
    def calculate_return(self, prices, years):
        """
        Calculate annualized return over specified period
        
        Args:
            prices (pd.Series): Price series
            years (int): Number of years to look back
            
        Returns:
            float: Annualized return as decimal (e.g., 0.15 for 15%)
        """
        if len(prices) < 2:
            return None
            
        # Get prices from years ago and current
        cutoff_date = prices.index[-1] - timedelta(days=years * 365)
        historical_prices = prices[prices.index >= cutoff_date]
        
        if len(historical_prices) < 2:
            return None
            
        initial_price = historical_prices.iloc[0]
        final_price = historical_prices.iloc[-1]
        
        if initial_price <= 0:
            return None
            
        # Calculate actual years between first and last price
        actual_years = (historical_prices.index[-1] - historical_prices.index[0]).days / 365.25
        
        if actual_years < years * 0.9:  # Allow 10% tolerance
            return None
            
        # Annualized return: (final/initial)^(1/years) - 1
        annualized_return = (final_price / initial_price) ** (1 / actual_years) - 1
        
        return annualized_return
        
    def calculate_annualized_std_dev(self, prices, asset_type='fund'):
        """
        Calculate annualized standard deviation (volatility)
        
        Args:
            prices (pd.Series): Price series
            asset_type (str): Type of asset for trading days adjustment
            
        Returns:
            float: Annualized standard deviation as decimal
        """
        if len(prices) < 2:
            return None
            
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        
        if len(returns) < 2:
            return None
            
        # Get trading days for this asset type
        trading_days = get_trading_days(asset_type)
        
        # Annualize: std(daily_returns) * sqrt(trading_days)
        annualized_std = returns.std() * np.sqrt(trading_days)
        
        return annualized_std
        
    def calculate_sharpe_ratio(self, prices, asset_type='fund'):
        """
        Calculate Sharpe ratio (risk-adjusted return)
        
        Args:
            prices (pd.Series): Price series
            asset_type (str): Type of asset for trading days adjustment
            
        Returns:
            float: Sharpe ratio
        """
        if len(prices) < 2:
            return None
            
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        
        if len(returns) < 2:
            return None
            
        # Get trading days for this asset type
        trading_days = get_trading_days(asset_type)
        
        # Annualized return
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        years = (prices.index[-1] - prices.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Annualized volatility
        annualized_std = returns.std() * np.sqrt(trading_days)
        
        if annualized_std == 0:
            return None
            
        # Sharpe ratio: (return - risk_free_rate) / volatility
        sharpe = (annualized_return - self.risk_free_rate) / annualized_std
        
        return sharpe
        
    def calculate_max_drawdown(self, prices):
        """
        Calculate maximum drawdown (largest peak-to-trough decline)
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            float: Maximum drawdown as decimal (negative value, e.g., -0.25 for -25%)
        """
        if len(prices) < 2:
            return None
            
        # Calculate running maximum
        running_max = prices.expanding().max()
        
        # Calculate drawdown at each point
        drawdown = (prices - running_max) / running_max
        
        # Maximum drawdown is the minimum value (most negative)
        max_dd = drawdown.min()
        
        return max_dd
        
    def calculate_max_drawdown_period(self, prices, years):
        """
        Calculate maximum drawdown over a specific period
        
        Args:
            prices (pd.Series): Price series
            years (int): Number of years to look back
            
        Returns:
            float: Maximum drawdown as decimal
        """
        if len(prices) < 2:
            return None
            
        # Get prices from years ago
        cutoff_date = prices.index[-1] - timedelta(days=years * 365)
        period_prices = prices[prices.index >= cutoff_date]
        
        if len(period_prices) < 2:
            return None
            
        return self.calculate_max_drawdown(period_prices)
        
    def calculate_rolling_cagr(self, prices, window_years):
        """
        Calculate rolling CAGR for specified window
        
        Args:
            prices (pd.Series): Price series
            window_years (int): Rolling window in years
            
        Returns:
            list: List of dicts with {date, cagr} for each window
        """
        if len(prices) < 2:
            return []
            
        results = []
        window_days = window_years * 365
        
        # Iterate through prices with rolling window
        for i in range(len(prices)):
            current_date = prices.index[i]
            
            # Get historical window
            start_date = current_date - timedelta(days=window_days)
            window_prices = prices[(prices.index >= start_date) & (prices.index <= current_date)]
            
            if len(window_prices) < 2:
                continue
                
            # Check if we have enough data (at least 90% of the window)
            actual_days = (window_prices.index[-1] - window_prices.index[0]).days
            if actual_days < window_days * 0.9:
                continue
                
            # Calculate CAGR
            initial_price = window_prices.iloc[0]
            final_price = window_prices.iloc[-1]
            
            if initial_price <= 0:
                continue
                
            actual_years = actual_days / 365.25
            cagr = (final_price / initial_price) ** (1 / actual_years) - 1
            
            results.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'cagr': float(cagr)
            })
            
        return results
        
    def calculate_all_metrics(self, asset_code, prices_df, asset_type='fund'):
        """
        Calculate all metrics for an asset
        
        Args:
            asset_code (str): Asset code
            prices_df (pd.DataFrame): DataFrame with 'date' and 'price' columns
            asset_type (str): Type of asset
            
        Returns:
            dict: Dictionary of all calculated metrics
        """
        # Convert to Series with datetime index
        prices_df = prices_df.copy()
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        prices_df = prices_df.sort_values('date')
        prices = prices_df.set_index('date')['price']
        
        metrics = {
            'asset_code': asset_code,
            'calculated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Calculate returns
        metrics['return_1y'] = self.calculate_return(prices, 1)
        metrics['return_3y_annualized'] = self.calculate_return(prices, 3)
        metrics['return_5y_annualized'] = self.calculate_return(prices, 5)
        
        # Calculate risk metrics
        metrics['std_dev_annualized'] = self.calculate_annualized_std_dev(prices, asset_type)
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(prices, asset_type)
        metrics['max_drawdown_3y'] = self.calculate_max_drawdown_period(prices, 3)
        metrics['max_drawdown_5y'] = self.calculate_max_drawdown_period(prices, 5)
        
        # Calculate rolling CAGR for all windows
        for window in ROLLING_WINDOWS:
            rolling_cagr = self.calculate_rolling_cagr(prices, window)
            # Store as JSON string
            metrics[f'rolling_cagr_{window}y'] = json.dumps(rolling_cagr) if rolling_cagr else None
            
        return metrics

def main():
    """Test the calculator with sample data"""
    import sqlite3
    from config import DB_PATH
    
    print("🧪 Testing Metrics Calculator\n")
    
    # Load sample asset
    conn = sqlite3.connect(DB_PATH)
    
    # Get first asset
    asset = pd.read_sql_query("SELECT asset_code, asset_type FROM assets LIMIT 1", conn).iloc[0]
    asset_code = asset['asset_code']
    asset_type = asset['asset_type']
    
    print(f"Testing with asset: {asset_code} (type: {asset_type})")
    
    # Get price data
    prices_df = pd.read_sql_query(
        f"SELECT date, price FROM price_data WHERE asset_code = '{asset_code}' ORDER BY date",
        conn
    )
    
    conn.close()
    
    print(f"Loaded {len(prices_df)} price records")
    print(f"Date range: {prices_df['date'].min()} to {prices_df['date'].max()}\n")
    
    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(asset_code, prices_df, asset_type)
    
    # Display results
    print("📊 Calculated Metrics:\n")
    print(f"1-Year Return: {metrics['return_1y']:.2%}" if metrics['return_1y'] else "1-Year Return: N/A")
    print(f"3-Year Return (p.a.): {metrics['return_3y_annualized']:.2%}" if metrics['return_3y_annualized'] else "3-Year Return: N/A")
    print(f"5-Year Return (p.a.): {metrics['return_5y_annualized']:.2%}" if metrics['return_5y_annualized'] else "5-Year Return: N/A")
    print(f"Volatility (annualized): {metrics['std_dev_annualized']:.2%}" if metrics['std_dev_annualized'] else "Volatility: N/A")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}" if metrics['sharpe_ratio'] else "Sharpe Ratio: N/A")
    print(f"Max Drawdown (3Y): {metrics['max_drawdown_3y']:.2%}" if metrics['max_drawdown_3y'] else "Max Drawdown (3Y): N/A")
    print(f"Max Drawdown (5Y): {metrics['max_drawdown_5y']:.2%}" if metrics['max_drawdown_5y'] else "Max Drawdown (5Y): N/A")
    
    # Show rolling CAGR sample
    for window in ROLLING_WINDOWS:
        key = f'rolling_cagr_{window}y'
        if metrics[key]:
            rolling_data = json.loads(metrics[key])
            print(f"\nRolling {window}Y CAGR: {len(rolling_data)} data points")
            if rolling_data:
                print(f"  Latest: {rolling_data[-1]['date']} = {rolling_data[-1]['cagr']:.2%}")

if __name__ == "__main__":
    main()
