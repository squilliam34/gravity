import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

def calculate_daily_market_cap(ticker: str, 
                                 start: str = '2000-01-01'
                                 ) -> pd.Series:
    """
    Calculate a series of daily market caps for a given stock using its ticker.

    Retrieves the most recent amount of shares, which adjusts for stock splits, etc over time.
    The historically adjusted prices also account for stock splits, so multiplying the current
    number of shares by the current amount of shares gives historic daily market caps. The 
    adjusted prices are resampled and averaged to the end of the month.

    Args:
    ticker (str): The stock symbol to retrieve the historic market caps for.
    start (str): The date to start calculating the market capitalization from in the format 
      YYYY-MM-DD.

    Returns:
    pd.Series: A daily series containing a company's market capitalization at different points 
      in time.
    """
    company = yf.Ticker(ticker)
    income_stmt = company.quarterly_income_stmt
    shares = income_stmt[income_stmt.columns[0]]['Basic Average Shares']
    prices = company.history(start = start)['Close']
    market_caps = (shares*prices)
    return market_caps

def create_market_cap_matrix(tickers: list[str]) -> np.ndarray:
    """
    Creates a TxN matrix of daily market caps, where T is the number
    of time periods, and N is the number of stocks in the universe.
    Market caps are scaled by a factor of 1e9 to avoid future calculation
    overflow.

    Args:
    tickers (list[str]): A list of stock tickers to populate the matrix with.

    Returns:
    np.ndarray: A 2D matrix containing market caps for N stocks across T periods of time.
    """
    # Need to put daily market caps in df first 
    # due to differences in lengths
    series_dict = {}
    for ticker in tickers:
        # Applies a log transformation to market caps to compress the space
        series_dict[ticker] = np.log(calculate_daily_market_cap(ticker))
        market_cap_matrix = pd.DataFrame(series_dict)
    return market_cap_matrix.to_numpy()

def create_date_range(start_date: str = '2000-01-01', 
                      end_date: str = date.today().strftime('%Y-%m-%d')
                      ) -> pd.DatetimeIndex:
    """
    Helper function to create an accurate range of dates.

    Args:
    start_date (str): The start date for the range.
    end_date (str): The end date for the range.

    Returns:
    pd.DatetimeIndex: An index of date ranges.
    """
    # Need to offset months by 1 to ensure lengths match
    today_pd = pd.to_datetime(end_date)
    next_month = (today_pd + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    return pd.date_range(start=start_date, end=next_month, freq='ME')

def calculate_market_cap_products(tickers: list[str], 
                                  start_date: str = '2000-01-01', 
                                  end_date: str = date.today().strftime('%Y-%m-%d')
                                  ) -> pd.DataFrame:
    """
    Calculate market cap products for every 2 given stocks across a given timeframe.

    Args:
    tickers (list[str]): A list of tickers in the universe whose masses will be multiplied to 
      populate the matrix.
    start_date (str): The start date of the range.
    end_date (str): The end date of the range.

    Returns:
    pd.DataFrame: A DataFrame that contains every mass product across every window 
      of time in the range of dates.
    """
    # Create range of dates
    dates = create_date_range(start_date=start_date, end_date=end_date)
    
    market_cap_matrix = create_market_cap_matrix(tickers)
    # Now calculate outerproducts between MASS_i and MASS_j
    mass_matrix = np.einsum('ti,tj->tij', market_cap_matrix, market_cap_matrix)

    # Convert back to df for easy access
    triu_index = np.triu_indices(len(tickers), 1)
    tickers_array = np.array(tickers)
    
    triu_i, triu_j = triu_index

    num_pairs = len(triu_i)  
    num_dates = len(dates)   

    return pd.DataFrame({
            'date': np.repeat(dates, num_pairs),
            'stock_i': np.tile(tickers_array[triu_i], num_dates),
            'stock_j': np.tile(tickers_array[triu_j], num_dates),
            'mass_i * mass_j': mass_matrix[:, triu_i, triu_j].reshape(-1)
    })