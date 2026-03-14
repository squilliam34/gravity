import yfinance as yf
from datetime import date
import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

def load_stock_data(ticker: str, start_date: str = '2000-01-01', end_date: str = date.today().strftime('%Y-%m-%d'), interval: str = '1d'):
    """
    Load historical stock price data for a given ticker symbol.

    Parameters:
    - ticker (str): The stock ticker symbol (e.g., 'NVDA').
    - start_date (str): The start date for the historical data in 'YYYY-MM-DD' format.
    - end_date (str): The end date for the historical data in 'YYYY-MM-DD' format.
    - interval (str): The data interval (e.g., '1d' for daily, '1wk' for weekly).

    Returns:
    - DataFrame: A DataFrame containing the historical stock price data.
    """
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date, interval=interval)
    stock_data.index = stock_data.index.date  
    return stock_data

def load_sp500_data(start_date: str = '2000-01-01', end_date: str = date.today().strftime('%Y-%m-%d'), interval: str = '1d'):
    """
    Load historical S&P 500 index data and calculate its daily yield.

    Parameters:
    - start_date (str): The start date for the historical data in 'YYYY-MM-DD' format.
    - end_date (str): The end date for the historical data in 'YYYY-MM-DD' format.
    - interval (str): The data interval (e.g., '1d' for daily, '1wk' for weekly).

    Returns:
    - DataFrame: A DataFrame containing the historical S&P 500 index data.
    """
    sp = load_stock_data('^GSPC', start_date, end_date, interval)
    sp = get_sp500_yield(sp)
    return sp

def get_sp500_yield(sp_data: pd.DataFrame):
    """
    Calculate the daily percentage change (yield) of the S&P 500 index.

    Parameters:
    - sp_data (DataFrame): The historical S&P 500 index data.

    Returns:
    - DataFrame: A DataFrame containing the S&P 500 index data with the daily percentage change (yield).
    """
    sp_data['Yield'] = sp_data['Close'].pct_change()
    return sp_data

def load_10_year_treasury_data():
    """"
    Load historical 10-year Treasury yield data from FRED and process it.

    Returns:
    - Series: A Series containing the historical 10-year Treasury yield data.
    """
    load_dotenv()
    fred_api_key = os.getenv('FRED')
    fred = Fred(api_key=fred_api_key)
    treasury_10 = fred.get_series('DGS10')
    treasury_10 = match_treasury_indices(load_sp500_data(), treasury_10)
    treasury_10 = calculate_treasury_diff(treasury_10)
    return treasury_10

def match_treasury_indices(sp_data: pd.DataFrame, treasury_10: pd.Series):
    """
    Match the indices of the 10-year Treasury yield data with the S&P 500 index data.

    Parameters:
    - sp_data (DataFrame): The historical S&P 500 index data.
    - treasury_10 (Series): The historical 10-year Treasury yield data.

    Returns:
    - DataFrame: A DataFrame containing the matched 10-year Treasury yield data.
    """
    treasury_10 = treasury_10[treasury_10.index.isin(sp_data.index)].to_frame(name='10Y_Treasury_Yield')
    return treasury_10

def calculate_treasury_diff(treasury_10: pd.DataFrame):
    """
    Process the 10-year Treasury yield data by calculating the daily difference 
    and match the indices with the S&P 500 index data.

    Parameters:
    - treasury_10 (DataFrame): The historical 10-year Treasury yield data.

    Returns:
    - DataFrame: A DataFrame containing the processed 10-year Treasury yield data.
    """
    treasury_10['diff'] = treasury_10['10Y_Treasury_Yield'].diff()
    return treasury_10