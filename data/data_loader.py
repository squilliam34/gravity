import yfinance as yf
from datetime import date

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