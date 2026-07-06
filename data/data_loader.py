import yfinance as yf
from datetime import date
import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd
import numpy as np

def get_tickers(FILEPATH: str) -> list[str]:
    """
    Extract tickers from a csv file of tickers.

    Args:
    FILENAME (str): The path to the file with tickers to extract.

    Returns: 
    list[str]: A list of tickers.
    """
    # Assume csv file has a column named 'Ticker' with the list of ticker symbols
    return pd.read_excel(FILEPATH)['Ticker'].tolist()

def load_prices(ticker: str, 
                start_date: str = '2000-01-01', 
                end_date: str = date.today().strftime('%Y-%m-%d'), 
                interval: str = '1d'
                ) -> pd.DataFrame:
    """
    Load historical stock price data for a given ticker symbol.

    Args:
    ticker (str): The stock ticker symbol (e.g., 'NVDA').
    start_date (str): The start date for the historical data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the historical data in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d' for daily, '1wk' for weekly).

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock price data.
    """
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date, interval=interval)

    # Check if the stock data exists (may not depending on IPO date)
    if stock_data.empty:
        return stock_data

    stock_data.index = stock_data.index.tz_localize(None)
    stock_data = stock_data.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])
    return stock_data

def calculate_stock_returns(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the daily percentage change (returns) of the stock.

    Args:
    stock_data (pd.DataFrame): The historical stock price data.

    Returns:
    pd.DataFrame: A DataFrame containing the original stock data with an additional column for daily returns.
    """
    stock_data['Returns'] = stock_data['Close'].pct_change()
    return stock_data

def load_sp500_data(start_date: str = '2000-01-01', 
                    end_date: str = date.today().strftime('%Y-%m-%d'), 
                    interval: str = '1d'
                    ) -> pd.DataFrame:
    """
    Load historical S&P 500 index data and calculate its daily yield.

    Args:
    start_date (str): The start date for the historical data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the historical data in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d' for daily, '1wk' for weekly).
    Returns:
    pd.DataFrame: A DataFrame containing the historical S&P 500 index data.
    """
    try:
        sp = load_prices('^GSPC', start_date, end_date, interval)
        sp = get_sp500_yield(sp)
        return sp
    except Exception as e:
        print(f"[load_sp500_data] failed: {e}")
        return pd.DataFrame()

def get_sp500_yield(sp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the daily percentage change (yield) of the S&P 500 index.

    Args:
    sp_data (pd.DataFrame): The historical S&P 500 index data.

    Returns:
    pd.DataFrame: A DataFrame containing the S&P 500 index data 
      with the daily percentage change (yield).
    """
    sp_data['Market Return'] = sp_data['Close'].pct_change()
    return sp_data

def load_10_year_treasury_data() -> pd.DataFrame:
    """"
    Load historical 10-year Treasury yield data from FRED and process it.

    Returns:
    pd.DataFrame: A DataFrame containing the historical 10-year Treasury yield data.
    """
    try:
        load_dotenv()
        fred_api_key = os.getenv('FRED')
        if not fred_api_key:
            raise RuntimeError('FRED API key not found in environment variables.')

        fred = Fred(api_key=fred_api_key)
        treasury_10 = fred.get_series('DGS10').to_frame(name='10Y_Treasury_Yield')
        treasury_10.index = pd.to_datetime(treasury_10.index)
        treasury_10 = calculate_treasury_diff(treasury_10)
        return treasury_10
    except Exception as e:
        print(f"[load_10_year_treasury_data] failed: {e}")
        return pd.DataFrame()

def calculate_treasury_diff(treasury_10: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 10-year Treasury yield data by calculating the daily difference 
    and match the indices with the S&P 500 index data.

    Args:
    treasury_10 (pd.DataFrame): The historical 10-year Treasury yield data.

    Returns:
    pd.DataFrame: A DataFrame containing the processed 10-year Treasury yield data.
    """
    treasury_10['Rate Change'] = treasury_10['10Y_Treasury_Yield'].diff()
    return treasury_10

def match_indices(treasury: pd.DataFrame, 
                  sp: pd.DataFrame, 
                  stock: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match the indices of treasury and S&P data with the indices of stock data.

    Args:
    treasury (pd.DataFrame): The historical 10-year Treasury yield data.
    sp (pd.DataFrame): The historical S&P 500 index data.
    stock (pd.DataFrame): The historical stock price data.
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the matched treasury and S&P data.
    """
    treasury = treasury[treasury.index.isin(stock.index)]
    sp = sp[sp.index.isin(stock.index)]
    return treasury, sp

def get_momentum_factor(prices: pd.DataFrame) -> pd.Series:
    """
    Calculates a momentum factor using the 12-1 month momentum approach. This entails
    calculating returns over the last 12 months (~ 252 trading days) and excluding the most 
    recent month of trading (~ 21 trading days). After the last 12-1 month returns are calculated
    for a given stock, the returns for all stocks available at a given point in time are
    ranked, and the average return of the last decile is subtracted from the average return
    of the first decile.

    Args: 
    prices (pd.DataFrame): A DataFrame of prices for various stocks to compute the momentum for.

    Returns:
    pd.Series: A series that contains the shared momentum factor across time.
    """
    momentum_returns = prices.pct_change(252-21).shift(21).dropna(how='all')
    def decile_spread(row):
        row = row.dropna()
        top = row[row >= row.quantile(0.9)].mean()
        bottom = row[row <= row.quantile(0.1)].mean()
        return top - bottom

    spread = momentum_returns.apply(decile_spread, axis=1)
    spread.name = 'Spread'
    return spread

def load_factor_data(tickers: list[str], 
                     start_date: str = '2000-01-01', 
                     end_date: str = date.today().strftime('%Y-%m-%d'), 
                     interval: str = '1d'
                     ) -> pd.DataFrame:
    """
    Load and merge historical stock price data, S&P 500 index data, and 10-year 
    Treasury yield data for a list of ticker symbols.

    Args:
    tickers (list[str]): A list of stock ticker symbols to load data for (e.g., ['NVDA', 'AAPL']).
    start_date (str): The start date for the historical data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the historical data in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d' for daily, '1wk' for weekly).

    turns:
    - DataFrame: The merged DataFrame for the stocks, S&P 500 index, and 10-year Treasury yield.
    """
    try:
        treasury = load_10_year_treasury_data()
        market = load_sp500_data()

        prices = []
        for ticker in tickers:
            # Retrieve raw price data
            stock_data = load_prices(ticker)
            if stock_data.empty:
                print(f'[load_merged_data] warning: no data for ticker {ticker}')
                continue
            prices.append(stock_data)

            # Calculate returns that will be used as the target variable
            returns = calculate_stock_returns(stock_data)
            returns.columns = ['Close', ticker]

        prices = pd.concat(prices, axis=1)

        # Need to calculate momentum factor using only the closing prices
        subset = prices['Close']
        momentum = get_momentum_factor(subset)
        momentum.name = 'Momentum'  

        # Drop 'Close' first to clean up memory
        prices = prices.drop(columns=['Close'])

        treasury, market = match_indices(treasury, market, prices)

        # Concatenate factors
        final = pd.concat([
            prices, 
            momentum,
            market.get('Market Return', pd.Series()), 
            treasury.get('Rate Change', pd.Series())
        ], axis=1)
        
        final.index = pd.to_datetime(final.index)
        final.index.strftime('%Y-%m-%d')
        final = final.sort_index().dropna(subset=['Momentum'])
        # For some reason, the first entry with the interest rate change is NaN so drop first row
        final.drop(final.index[0], inplace=True)
        return final
    except Exception as e:
        print(f"[load_merged_data] failed: {e}")
        return pd.DataFrame()
