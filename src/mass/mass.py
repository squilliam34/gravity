import yfinance as yf
import pandas as pd
import numpy as np

def calculate_monthly_market_cap(ticker: str, 
                                 start: str = '2000-01-01'
                                 ) -> pd.Series:
    """
    Calculate a series of monthly market caps for a given stock using its ticker.

    Retrieves the most recent amount of shares, which adjusts for stock splits, etc over time.
    The historically adjusted prices also account for stock splits, so multiplying the current
    number of shares by the current amount of shares gives historic monthly market caps. The 
    adjusted prices are resampled and averaged to the end of the month.

    Parameters:
    - ticker (str): The stock symbol to retrieve the historic market caps for.
    - start (str): The date to start calculating the market capitalization from in the format
    YYYY-MM-DD.

    Returns:
    - pd.Series: A monthly series containing a company's market capitalization at different 
    points in time.
    """
    company = yf.Ticker(ticker)
    income_stmt = company.quarterly_income_stmt
    shares = company.income_stmt[income_stmt.columns[0]]['Basic Average Shares']
    prices = company.history(start = start)['Close']
    market_caps = (shares*prices).resample('ME').mean()
    return market_caps

def create_market_cap_matrix(tickers: list[str]) -> np.ndarray:
    """
    Creates a TxN matrix of monthly market caps, where T is the number
    of time periods, and N is the number of stocks in the universe.
    Market caps are scaled by a factor of 1e9 to avoid future calculation
    overflow

    Parameters:
    - tickers (list[str]): A list of stock tickers to populate the matrix with

    Returns:
    - np.ndarray: A 2D matrix containing market caps for N stocks across T periods
    of time
    """
    # Need to put monthly market caps in df first 
    # due to differences in lengths
    series_dict = {}
    for ticker in tickers:
        # Scales market caps by a factor of 1B to prevent overflow
        s = calculate_monthly_market_cap(ticker) / 1e9
        series_dict[ticker] = s
        market_cap_matrix = pd.DataFrame(series_dict)
    return market_cap_matrix.to_numpy()