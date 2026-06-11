import yfinance as yf
from datetime import date
import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

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
    try:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date, end=end_date, interval=interval)
        stock_data.index = stock_data.index.date
        stock_data = stock_data.drop(columns=['Open', 
                                              'High', 
                                              'Low', 
                                              'Volume', 
                                              'Dividends', 
                                              'Stock Splits'])
        return stock_data
    except Exception as e:
        print(f"[load_prices] failed for {ticker}: {e}")
        return pd.DataFrame()

def calculate_stock_returns(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the daily percentage change (returns) of the stock.

    Args:
    stock_data (DataFrame): The historical stock price data.

    Returns:
    DataFrame: A DataFrame containing the original stock data with an additional column for daily returns.
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
    sp_data (DataFrame): The historical S&P 500 index data.

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
    - DataFrame: A DataFrame containing the historical 10-year Treasury yield data.
    """
    try:
        load_dotenv()
        fred_api_key = os.getenv('FRED')
        if not fred_api_key:
            raise RuntimeError('FRED API key not found in environment variables.')

        fred = Fred(api_key=fred_api_key)
        treasury_10 = fred.get_series('DGS10').to_frame(name='10Y_Treasury_Yield')
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
    treasury_10 (DataFrame): The historical 10-year Treasury yield data.

    Returns:
    DataFrame: A DataFrame containing the processed 10-year Treasury yield data.
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
    treasury (DataFrame): The historical 10-year Treasury yield data.
    - sp (DataFrame): The historical S&P 500 index data.
    - stock (DataFrame): The historical stock price data.
   Returns:
    - Tuple[DataFrame, DataFrame]: A tuple containing the matched treasury and S&P data.
    """
    treasury = treasury[treasury.index.isin(stock.index)]
    sp = sp[sp.index.isin(stock.index)]
    return treasury, sp

def load_factor_data(tickers: list[str], 
                     start_date: str = '2000-01-01', 
                     end_date: str = date.today().strftime('%Y-%m-%d'), 
                     interval: str = '1d'
                     ) -> pd.DataFrame:
    """
    Load and merge historical stock price data, S&P 500 index data, and 10-year 
    Treasury yield data for a list of ticker symbols.

    Args:
    tickers (list[str]): A list of stock ticker symbols to load data for 
    (e.g., ['NVDA', 'AAPL']).

    turns:
    - DataFrame: The merged DataFrame for the stocks, S&P 500 index, and 10-year Treasury yield.
    """
    try:
        treasury = load_10_year_treasury_data()
        sp = load_sp500_data(start_date, end_date, interval)

        stock_data_frames = []
        valid_tickers = []
        for ticker in tickers:
            stock_data = load_stock_data(ticker, start_date, end_date, interval)
            if stock_data.empty:
                print(f'[load_merged_data] warning: no data for ticker {ticker}')
                continue
            stock_data_frames.append(stock_data)
            valid_tickers.append(ticker)

        if not stock_data_frames:
            return pd.DataFrame()

        merged_data = pd.concat(stock_data_frames, axis=1, keys=valid_tickers)

        treasury, sp = match_indices(treasury, sp, merged_data)

        final = pd.concat([merged_data, 
                           sp.get('Market Return', pd.Series()), 
                           treasury.get('Rate Change', pd.Series())], 
                           axis=1)
        final.index = pd.to_datetime(final.index)
        final.index.strftime('%Y-%m-%d')
        return final
    except Exception as e:
        print(f"[load_merged_data] failed: {e}")
        return pd.DataFrame()
