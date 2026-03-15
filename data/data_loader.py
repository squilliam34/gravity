import yfinance as yf
from datetime import date
import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

def load_prices(ticker: str, 
                start_date: str = '2000-01-01', 
                end_date: str = date.today().strftime('%Y-%m-%d'), 
                interval: str = '1d'
                ) -> pd.DataFrame:
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

def load_stock_data(ticker: str, 
                    start_date: str = '2000-01-01', 
                    end_date: str = date.today().strftime('%Y-%m-%d'), 
                    interval: str = '1d'
                    ) -> pd.DataFrame:
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
    try:
        stock_data = load_prices(ticker, start_date, end_date, interval)
        stock_data = calculate_20_day_ma(stock_data)
        stock_data = calculate_momentum(stock_data)
        stock_data.dropna(inplace=True)
        stock_data = calculate_stock_returns(stock_data)
        stock_data = stock_data.drop(columns=['Close'])
        return stock_data
    except Exception as e:
        print(f"[load_stock_data] failed for {ticker}: {e}")
        return pd.DataFrame()

def calculate_20_day_ma(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the 20-day moving average for the given stock data.

    Parameters:
    - stock_data (DataFrame): The historical stock price data.

    Returns:
    - DataFrame: A DataFrame containing the original stock data with an 
    additional column for the 20-day moving average.
    """
    stock_data['20_day_MA'] = stock_data['Close'].rolling(window=20).mean()
    return stock_data

def calculate_momentum(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the momentum of the stock based on the 20-day moving average.
    Use the previous day's closing price and the previous day's 20-day moving 
    average to calculate momentum so that the factor model isn't using future information.

    Parameters:
    - stock_data (DataFrame): The historical stock price data with the 20-day moving average.

    Returns:
    - DataFrame: A DataFrame containing the original stock data with an additional column for momentum.
    """
    close_prev = stock_data['Close'].shift(1)
    ma_prev = stock_data['20_day_MA'].shift(1)

    stock_data['Momentum'] = (close_prev - ma_prev) / close_prev
    return stock_data

def calculate_stock_returns(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the daily percentage change (returns) of the stock.

    Parameters:
    - stock_data (DataFrame): The historical stock price data.

    Returns:
    - DataFrame: A DataFrame containing the original stock data with an additional column for daily returns.
    """
    stock_data['Returns'] = stock_data['Close'].pct_change()
    return stock_data

def load_sp500_data(start_date: str = '2000-01-01', 
                    end_date: str = date.today().strftime('%Y-%m-%d'), 
                    interval: str = '1d'
                    ) -> pd.DataFrame:
    """
    Load historical S&P 500 index data and calculate its daily yield.

    Parameters:
    - start_date (str): The start date for the historical data in 'YYYY-MM-DD' format.
    - end_date (str): The end date for the historical data in 'YYYY-MM-DD' format.
    - interval (str): The data interval (e.g., '1d' for daily, '1wk' for weekly).

    Returns:
    - DataFrame: A DataFrame containing the historical S&P 500 index data.
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

    Parameters:
    - sp_data (DataFrame): The historical S&P 500 index data.

    Returns:
    - DataFrame: A DataFrame containing the S&P 500 index data 
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

    Parameters:
    - treasury_10 (DataFrame): The historical 10-year Treasury yield data.

    Returns:
    - DataFrame: A DataFrame containing the processed 10-year Treasury yield data.
    """
    treasury_10['Rate Change'] = treasury_10['10Y_Treasury_Yield'].diff()
    return treasury_10

def match_indices(treasury: pd.DataFrame, 
                  sp: pd.DataFrame, 
                  stock: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match the indices of treasury and S&P data with the indices of stock data.

    Parameters:
    - treasury (DataFrame): The historical 10-year Treasury yield data.
    - sp (DataFrame): The historical S&P 500 index data.
    - stock (DataFrame): The historical stock price data.

    Returns:
    - Tuple[DataFrame, DataFrame]: A tuple containing the matched treasury and S&P data.
    """
    treasury = treasury[treasury.index.isin(stock.index)]
    sp = sp[sp.index.isin(stock.index)]
    return treasury, sp

def load_merged_data(tickers: list[str], 
                     start_date: str = '2000-01-01', 
                     end_date: str = date.today().strftime('%Y-%m-%d'), 
                     interval: str = '1d'
                     ) -> pd.DataFrame:
    """
    Load and merge historical stock price data, S&P 500 index data, and 10-year 
    Treasury yield data for a list of ticker symbols.

    Parameters:
    - tickers (list[str]): A list of stock ticker symbols to load data for 
    (e.g., ['NVDA', 'AAPL']).

    Returns:
    - DataFrame: A DataFrame containing the merged data for the stocks, S&P 500 index, 
    and 10-year Treasury yield.
    """
    try:
        treasury = load_10_year_treasury_data()
        sp = load_sp500_data(start_date, end_date, interval)

        stock_data_frames = []
        for ticker in tickers:
            stock_data = load_stock_data(ticker, start_date, end_date, interval)
            if stock_data.empty:
                print(f"[load_merged_data] warning: no data for ticker {ticker}")
            stock_data_frames.append(stock_data)

        if not stock_data_frames:
            return pd.DataFrame()

        merged_data = pd.concat(stock_data_frames, axis=1, keys=tickers)
        merged_data = merged_data.dropna()

        treasury, sp = match_indices(treasury, sp, merged_data)

        final = pd.concat([merged_data, 
                           sp.get('Market Return', pd.Series()), 
                           treasury.get('Rate Change', pd.Series())], 
                           axis=1).dropna()
        return final
    except Exception as e:
        print(f"[load_merged_data] failed: {e}")
        return pd.DataFrame()
