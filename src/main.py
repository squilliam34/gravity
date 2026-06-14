from data.data_loader import get_tickers
from src.mass.mass import calculate_market_cap_products
from src.distance.distance import build_distances
import pandas as pd
from datetime import date

def calculate_gravity(tickers: list[str], 
                      start_date: str = '2000-01-01', 
                      end_date: str = date.today().strftime('%Y-%m-%d'), 
                      interval: str = '1d',
                      window: int = 252) -> pd.DataFrame:
    """
    Calculates a measure of gravity between stocks of the form M1*M2/D.

    Args:
    tickers (list[str]): A list of tickers to calculate the gravities between.
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d' for daily, '1wk' for weekly).
    window (int): The size of the rolling window to use for the factor model
      (default is 252 trading days, approximately one year).

    Returns:
    pd.DataFrame: A DataFrame containing the mass, distance, and gravity between
      stock combinations over time.
    """
    distance = build_distances(tickers=tickers, 
                               start_date=start_date, 
                               end_date=end_date, 
                               interval=interval,
                               window=window)
    mass = calculate_market_cap_products(tickers=tickers,
                                         start_date=start_date,
                                         end_date=end_date)

    # Need to convert index of mass to datetime
    mass = mass.reset_index()
    mass['date'] = pd.to_datetime(mass['date'])
    mass = mass.set_index(['date', 'stock_i', 'stock_j'])

    # Subet mass to same index as distances
    dates = distance.index.get_level_values('date').unique()
    mass = mass.loc[
        mass.index.get_level_values('date').isin(dates)
    ]

    # Join distance and mass for easy calculation
    mass = mass.join(distance, on=['date', 'stock_i', 'stock_j'])

    # Calculate gravity
    mass['Gravity'] = mass['mass_i * mass_j'] / mass['Distance']
    
    return mass