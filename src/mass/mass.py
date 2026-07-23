import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from pandas.tseries.offsets import Day

def calculate_daily_market_cap(ticker: str, 
                               start_date: str = '2000-01-01',
                               end_date: str = date.today().strftime('%Y-%m-%d')
                              ) -> pd.Series:
    """
    Calculate a series of daily market caps for a given stock using its ticker.

    Retrieves the most recent amount of shares, which adjusts for stock splits, etc over time.
    The historically adjusted prices also account for stock splits, so multiplying the current
    number of shares by the current amount of shares gives historic daily market caps. The 
    adjusted prices are resampled and averaged to the end of the month.

    Args:
    ticker (str): The stock symbol to retrieve the historic market caps for.
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.
    
    Returns:
    pd.Series: A daily series containing a company's market capitalization at different 
    points in time.
    """
    company = yf.Ticker(ticker)
    income_stmt = company.quarterly_income_stmt
    shares = income_stmt[income_stmt.columns[0]]['Basic Average Shares']

    # Need to include extra market cap data for rolling averages
    history_start = (
        pd.Timestamp(start_date)
        - Day(45)
    ).strftime('%Y-%m-%d')
    prices = company.history(start = history_start, end = end_date)['Close']
    
    # Log market caps to compress space
    market_caps = np.log((shares*prices))
    market_caps = market_caps.rename(ticker)
    return market_caps

def create_market_cap_df(tickers:list[str],
                         start_date: str = '2000-01-01',
                         end_date: str = date.today().strftime('%Y-%m-%d')
                        ) -> pd.DataFrame:
    """
    Args:
    tickers (list[str]): A list of tickers to retrieve market caps for.
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.
    
    Returns:
    pd.DataFrame: A DataFrame containing daily series of a company's market capitalization.
    """
    mkt_cps = [calculate_daily_market_cap(ticker) for ticker in tickers]
    mkt_cps = pd.concat(mkt_cps, axis = 1)
    mkt_cps = (
        mkt_cps
        .stack()
        .rename('market_cap')
    )

    # Need to convert date to match format of other objects to be used
    mkt_cps.index.names = ['date', 'ticker']
    mkt_cps = mkt_cps.reset_index()
    mkt_cps['date'] = pd.to_datetime(mkt_cps['date']).dt.tz_localize(None)
    mkt_cps = mkt_cps.set_index(['date','ticker'])

    # Calculate rolling period monthly average
    mkt_cps = (
        mkt_cps
        .groupby(level='ticker')['market_cap']
        .rolling(window=21, min_periods=21)
        .mean()
        .droplevel(0)
        .to_frame()
    )
    # Trim back to desired start date
    mkt_cps = mkt_cps.loc[
        mkt_cps.index.get_level_values('date') >= pd.Timestamp(start_date)
    ]

    # Downsample data to only include first trading day of each month
    dates = mkt_cps.index.get_level_values('date')
    mkt_cps = (
        mkt_cps
        .assign(month=dates.to_period('M'))
        .reset_index()
        .sort_values('date')
        .groupby(['month', 'ticker'], as_index=False)
        .first()
        .drop(columns='month')
        .set_index(['date', 'ticker'])
        .sort_index()
    )

    return mkt_cps