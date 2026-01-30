import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3

class MetricsCalculator:
    def __init__(self, db_path="db/investment_data.db"):
        self.db_path = db_path

    def get_returns_data(self, asset_codes):
        """Get price data and calculate returns for assets"""
        conn = sqlite3.connect(self.db_path)

        placeholders = ','.join(['?' for _ in asset_codes])
        query = f'''
            SELECT asset_code, date, price
            FROM price_data
            WHERE asset_code IN ({placeholders})
            ORDER BY asset_code, date
        '''

        df = pd.read_sql_query(query, conn, params=asset_codes)
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

        # CRITICAL FIX: Normalize all dates to timezone-naive to prevent comparison errors
        # Different data sources (VN funds, crypto, US stocks) may have different timezone handling
        # Force conversion to timezone-naive by using tz_localize(None) which removes timezone info
        try:
            # First, ensure all dates are datetime objects
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], utc=True)
            
            # Check if the series has timezone info
            if df['date'].dt.tz is not None:
                # Convert to UTC first, then remove timezone
                df['date'] = df['date'].dt.tz_convert('UTC').dt.tz_localize(None)
            else:
                # Try to remove timezone anyway (handles mixed cases)
                try:
                    df['date'] = pd.to_datetime(df['date'].astype(str)).dt.tz_localize(None)
                except:
                    # Already timezone-naive
                    pass
        except Exception as e:
            # Last resort: convert to string and back to remove any timezone info
            try:
                df['date'] = pd.to_datetime(df['date'].astype(str))
            except:
                pass
        
        # Pivot to get assets as columns
        price_data = df.pivot(index='date', columns='asset_code', values='price')

        # Calculate daily returns for each asset separately to handle different reporting schedules
        returns_data = price_data.pct_change(fill_method=None)

        # Only drop rows where ALL assets are NaN (keep rows with partial data)
        returns_data = returns_data.dropna(how='all')

        return price_data, returns_data

    def calculate_rolling_cagr(self, price_data, years=4):
        """Calculate rolling CAGR for specified number of years"""
        rolling_cagr = pd.DataFrame()

        for asset in price_data.columns:
            prices = price_data[asset].dropna()
            if len(prices) == 0:
                continue

            cagr_series = pd.Series(index=prices.index, dtype=float)

            for i in range(len(prices)):
                end_date = prices.index[i]
                start_date = end_date - timedelta(days=years*365)

                # Find the closest start price
                start_prices = prices[prices.index <= start_date]
                if len(start_prices) == 0:
                    cagr_series.iloc[i] = np.nan
                    continue

                start_price = start_prices.iloc[-1]
                end_price = prices.iloc[i]

                # Calculate actual time difference in years
                actual_years = (end_date - start_prices.index[-1]).days / 365.25

                if actual_years < years - 0.1:  # Not enough data
                    cagr_series.iloc[i] = np.nan
                else:
                    cagr = ((end_price / start_price) ** (1/actual_years)) - 1
                    cagr_series.iloc[i] = cagr * 100  # Convert to percentage

            rolling_cagr[asset] = cagr_series

        return rolling_cagr.dropna(how='all')

    def calculate_total_return(self, price_data, start_date=None, end_date=None):
        """Calculate total return for each asset over specified period"""
        filtered_data = price_data.copy()

        if start_date:
            filtered_data = filtered_data[filtered_data.index >= start_date]
        if end_date:
            filtered_data = filtered_data[filtered_data.index <= end_date]

        total_returns = {}
        for asset in filtered_data.columns:
            prices = filtered_data[asset].dropna()
            if len(prices) > 0:
                total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
                total_returns[asset] = total_return

        return total_returns

    def calculate_annualized_return(self, price_data, start_date=None, end_date=None):
        """Calculate annualized return (CAGR) for each asset"""
        filtered_data = price_data.copy()

        if start_date:
            filtered_data = filtered_data[filtered_data.index >= start_date]
        if end_date:
            filtered_data = filtered_data[filtered_data.index <= end_date]

        annualized_returns = {}
        for asset in filtered_data.columns:
            prices = filtered_data[asset].dropna()
            if len(prices) > 1:
                years = (prices.index[-1] - prices.index[0]).days / 365.25
                if years > 0:
                    cagr = ((prices.iloc[-1] / prices.iloc[0]) ** (1/years)) - 1
                    annualized_returns[asset] = cagr * 100

        return annualized_returns

    def calculate_volatility(self, returns_data, annualized=True, frequency='daily'):
        """Calculate volatility (standard deviation of returns)"""
        volatility = returns_data.std()

        if annualized:
            if frequency == 'daily':
                factor = np.sqrt(252)  # 252 trading days in a year
            elif frequency == 'monthly':
                factor = np.sqrt(12)   # 12 months in a year
            else:
                factor = 1
            
            volatility = volatility * factor

        return volatility * 100  # Convert to percentage

    def calculate_sharpe_ratio(self, returns_data, risk_free_rate=0.02, min_periods=30):
        """Calculate Sharpe ratio (assuming 2% risk-free rate)

        Args:
            returns_data: DataFrame of returns
            risk_free_rate: Annual risk-free rate (default 2%)
            min_periods: Minimum number of data points required for reliable calculation
        """
        sharpe_ratios = {}

        for asset in returns_data.columns:
            asset_returns = returns_data[asset].dropna()

            # Require minimum data points for reliable calculation
            if len(asset_returns) < min_periods:
                sharpe_ratios[asset] = np.nan
                continue

            # Calculate annualized returns and volatility for this asset
            mean_return = asset_returns.mean() * 252  # Annualized (decimal form)
            volatility = asset_returns.std() * np.sqrt(252)  # Annualized (decimal form)

            if volatility == 0:
                sharpe_ratios[asset] = np.nan
            else:
                sharpe_ratios[asset] = (mean_return - risk_free_rate) / volatility

        return pd.Series(sharpe_ratios)

    def get_month_end_returns_data(self, asset_codes, start_date=None, end_date=None):
        """
        Get month-end price data and calculate returns for assets.
        This standardizes the data frequency across all assets to avoid noise
        from different NAV reporting frequencies.
        
        Args:
            asset_codes: List of asset codes
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Tuple of (price_data, returns_data) DataFrames with month-end data
        """
        from data_processor import DataProcessor
        
        processor = DataProcessor(self.db_path)
        df = processor.get_month_end_price_data(asset_codes, start_date, end_date)
        
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Pivot to get assets as columns
        price_data = df.pivot(index='date', columns='asset_code', values='price')
        
        # Calculate monthly returns for each asset
        returns_data = price_data.pct_change(fill_method=None)
        
        # Only drop rows where ALL assets are NaN
        returns_data = returns_data.dropna(how='all')
        
        return price_data, returns_data

    def calculate_sharpe_ratio_month_end(self, asset_codes, risk_free_rate=0.02, min_periods=12, 
                                         start_date=None, end_date=None):
        """
        Calculate Sharpe ratio using month-end data for standardization.
        
        This method addresses the issue where different assets have different numbers
        of NAV data points, which creates noise in the Sharpe ratio calculation.
        By using month-end data, we ensure:
        1. All assets are compared on the same frequency (monthly)
        2. The calculation is less affected by different reporting schedules
        3. Results are more stable and comparable across assets
        
        Args:
            asset_codes: List of asset codes
            risk_free_rate: Annual risk-free rate (default 2%)
            min_periods: Minimum number of months required (default 12 months)
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            pd.Series: Sharpe ratios for each asset
        """
        _, returns_data = self.get_month_end_returns_data(asset_codes, start_date, end_date)
        
        if returns_data.empty:
            return pd.Series({asset: np.nan for asset in asset_codes})
        
        sharpe_ratios = {}
        
        for asset in asset_codes:
            if asset not in returns_data.columns:
                sharpe_ratios[asset] = np.nan
                continue
                
            asset_returns = returns_data[asset].dropna()
            
            # Require minimum data points for reliable calculation
            if len(asset_returns) < min_periods:
                sharpe_ratios[asset] = np.nan
                continue
            
            # Calculate annualized returns and volatility from monthly data
            # Monthly returns need to be annualized by multiplying by 12 (not 252)
            mean_return = asset_returns.mean() * 12  # Annualized from monthly (decimal form)
            volatility = asset_returns.std() * np.sqrt(12)  # Annualized from monthly (decimal form)
            
            if volatility == 0:
                sharpe_ratios[asset] = np.nan
            else:
                sharpe_ratios[asset] = (mean_return - risk_free_rate) / volatility
        
        return pd.Series(sharpe_ratios)


    def calculate_maximum_drawdown(self, price_data):
        """Calculate maximum drawdown for each asset"""
        drawdowns = {}

        for asset in price_data.columns:
            prices = price_data[asset].dropna()
            if len(prices) > 0:
                # Calculate running maximum
                peak = prices.expanding(min_periods=1).max()
                # Calculate drawdown
                dd = (prices - peak) / peak
                max_dd = dd.min() * 100  # Convert to percentage
                drawdowns[asset] = max_dd

        return drawdowns

    def calculate_rolling_volatility(self, returns_data, window=252):
        """Calculate rolling volatility"""
        rolling_vol = returns_data.rolling(window=window).std() * np.sqrt(252) * 100
        return rolling_vol

    def calculate_rolling_sharpe(self, returns_data, window=252, risk_free_rate=0.02):
        """Calculate rolling Sharpe ratio"""
        rolling_mean = returns_data.rolling(window=window).mean() * 252
        rolling_std = returns_data.rolling(window=window).std() * np.sqrt(252)

        rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
        return rolling_sharpe

    def calculate_correlation_matrix(self, returns_data):
        """Calculate correlation matrix between assets"""
        return returns_data.corr()

    def calculate_cagr_table(self, asset_codes, end_date=None):
        """
        Calculate CAGR table for multiple time periods (1-15 years) up to a specific end date.
        
        Args:
            asset_codes: List of asset codes to analyze
            end_date: End date for CAGR calculation (defaults to latest available date)
            
        Returns:
            DataFrame with assets as rows and time periods as columns
        """
        price_data, _ = self.get_returns_data(asset_codes)
        
        if price_data.empty:
            return pd.DataFrame()
        
        # Use latest date if end_date not specified
        if end_date is None:
            end_date = price_data.index.max()
        else:
            # Convert to pandas datetime if needed
            if not isinstance(end_date, pd.Timestamp):
                end_date = pd.to_datetime(end_date)
        
        # Filter data up to end_date
        price_data = price_data[price_data.index <= end_date]
        
        if price_data.empty:
            return pd.DataFrame()
        
        # Time periods to calculate (1-15 years)
        periods = list(range(1, 16))
        
        # Initialize results dictionary
        results = {asset: {} for asset in asset_codes}
        
        for asset in asset_codes:
            prices = price_data[asset].dropna()
            
            if len(prices) == 0:
                continue
            
            # Get the end price (closest to end_date)
            end_price = prices.iloc[-1]
            actual_end_date = prices.index[-1]
            
            for years in periods:
                # Calculate start date for this period
                target_start_date = actual_end_date - timedelta(days=years*365)
                
                # Find the closest start price
                start_prices = prices[prices.index <= target_start_date]
                
                if len(start_prices) == 0:
                    # Not enough data for this period
                    results[asset][f'{years}Y'] = None
                    continue
                
                start_price = start_prices.iloc[-1]
                actual_start_date = start_prices.index[-1]
                
                # Calculate actual time difference in years
                actual_years = (actual_end_date - actual_start_date).days / 365.25
                
                # Only calculate if we have at least 90% of the required period
                if actual_years < years * 0.9:
                    results[asset][f'{years}Y'] = None
                else:
                    # Calculate CAGR
                    cagr = ((end_price / start_price) ** (1/actual_years)) - 1
                    results[asset][f'{years}Y'] = cagr * 100  # Convert to percentage
        
        # Convert to DataFrame
        df = pd.DataFrame(results).T
        
        # Reorder columns to be 1Y, 2Y, 3Y, etc.
        column_order = [f'{i}Y' for i in periods]
        df = df[column_order]
        
        return df

    def get_performance_summary(self, asset_codes, periods=['3Y', '5Y', 'Inception']):
        """Get comprehensive performance summary"""
        price_data, returns_data = self.get_returns_data(asset_codes)

        summary = {}
        current_date = price_data.index.max()

        for period in periods:
            # Determine start date based on period
            if period == 'Inception':
                start_date = None
            else:
                years = int(period[0])
                start_date = current_date - timedelta(days=years*365)
            
            # IMPORTANT: Process each asset INDEPENDENTLY to avoid data loss
            # When multiple assets have different date ranges, combining them
            # can result in missing data for assets with longer histories
            
            total_returns = {}
            annualized_returns = {}
            volatilities = {}
            sharpe_ratios = {}
            risk_free_rate = 0.02
            
            for asset in asset_codes:
                # Get month-end data for THIS asset only
                me_price_data, me_returns_data = self.get_month_end_returns_data(
                    [asset],  # Single asset
                    start_date=start_date, 
                    end_date=current_date
                )
                
                # Calculate metrics for this asset
                if not me_price_data.empty:
                    # Total return
                    total_ret = self.calculate_total_return(me_price_data)
                    if asset in total_ret:
                        total_returns[asset] = total_ret[asset]
                    
                    # Annualized return
                    ann_ret = self.calculate_annualized_return(me_price_data)
                    if asset in ann_ret:
                        annualized_returns[asset] = ann_ret[asset]
                
                if not me_returns_data.empty and asset in me_returns_data.columns:
                    # Volatility
                    vol_series = self.calculate_volatility(me_returns_data[[asset]], frequency='monthly')
                    if asset in vol_series.index:
                        volatilities[asset] = vol_series[asset]
                    
                    # Sharpe ratio
                    asset_returns = me_returns_data[asset].dropna()
                    if len(asset_returns) >= 2:
                        mean_return = asset_returns.mean() * 12
                        vol = asset_returns.std() * np.sqrt(12)
                        
                        if vol > 0:
                            sharpe_ratios[asset] = (mean_return - risk_free_rate) / vol
                        else:
                            sharpe_ratios[asset] = np.nan
                    else:
                        sharpe_ratios[asset] = np.nan
            
            period_summary = {
                'total_return': total_returns,
                'annualized_return': annualized_returns,
                'volatility': volatilities,
                'sharpe_ratio': sharpe_ratios
            }
            summary[period] = period_summary

        # Add overall metrics (inception period)
        summary['max_drawdown'] = self.calculate_maximum_drawdown(price_data)
        summary['correlation'] = self.calculate_correlation_matrix(returns_data)

        return summary

if __name__ == "__main__":
    calc = MetricsCalculator()

    # Test with sample assets
    assets = ['DCDS', 'MAGEF', 'VNINDEX']
    summary = calc.get_performance_summary(assets)

    print("Performance Summary:")
    for period, data in summary.items():
        if period != 'correlation':
            print(f"\n{period}:")
            if isinstance(data, dict) and 'annualized_return' in data:
                print("Annualized Returns:", data['annualized_return'])
                print("Volatility:", data['volatility'])