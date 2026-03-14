import yfinance as yf
from datetime import date
import os
from dotenv import load_dotenv
from fredapi import Fred

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
    Load historical S&P 500 index data.

    Parameters:
    - start_date (str): The start date for the historical data in 'YYYY-MM-DD' format.
    - end_date (str): The end date for the historical data in 'YYYY-MM-DD' format.
    - interval (str): The data interval (e.g., '1d' for daily, '1wk' for weekly).

    Returns:
    - DataFrame: A DataFrame containing the historical S&P 500 index data.
    """
    sp = load_stock_data('^GSPC', start_date, end_date, interval)
    return sp

def load_10_year_treasury_data():
    """"
    Load historical 10-year Treasury yield data from FRED.

    Returns:
    - Series: A Series containing the historical 10-year Treasury yield data.
    """
    load_dotenv()
    fred_api_key = os.getenv('FRED')
    fred = Fred(api_key=fred_api_key)
    treasury_10 = fred.get_series('DGS10')
    return treasury_10