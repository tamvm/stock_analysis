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

        df['date'] = pd.to_datetime(df['date'])

        # Pivot to get assets as columns
        price_data = df.pivot(index='date', columns='asset_code', values='price')

        # Calculate daily returns
        returns_data = price_data.pct_change(fill_method=None).dropna()

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

    def calculate_volatility(self, returns_data, annualized=True):
        """Calculate volatility (standard deviation of returns)"""
        volatility = returns_data.std()

        if annualized:
            volatility = volatility * np.sqrt(252)  # 252 trading days in a year

        return volatility * 100  # Convert to percentage

    def calculate_sharpe_ratio(self, returns_data, risk_free_rate=0.02):
        """Calculate Sharpe ratio (assuming 2% risk-free rate)"""
        mean_returns = returns_data.mean() * 252  # Annualized
        volatility = returns_data.std() * np.sqrt(252)  # Annualized

        sharpe_ratios = (mean_returns - risk_free_rate) / volatility
        return sharpe_ratios

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

    def get_performance_summary(self, asset_codes, periods=['3Y', '5Y', 'Inception']):
        """Get comprehensive performance summary"""
        price_data, returns_data = self.get_returns_data(asset_codes)

        summary = {}
        current_date = price_data.index.max()

        for period in periods:
            if period == 'Inception':
                start_date = None
            else:
                years = int(period[0])
                start_date = current_date - timedelta(days=years*365)

            period_summary = {
                'total_return': self.calculate_total_return(price_data, start_date),
                'annualized_return': self.calculate_annualized_return(price_data, start_date),
                'volatility': self.calculate_volatility(
                    returns_data[returns_data.index >= (start_date or returns_data.index.min())]
                ).to_dict() if start_date else self.calculate_volatility(returns_data).to_dict()
            }
            summary[period] = period_summary

        # Add overall metrics
        summary['max_drawdown'] = self.calculate_maximum_drawdown(price_data)
        summary['sharpe_ratio'] = self.calculate_sharpe_ratio(returns_data).to_dict()
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