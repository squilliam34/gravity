import yfinance as yf
import pandas as pd

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