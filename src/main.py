import pandas as pd

def calculate_gravity(start_year: int) -> pd.Series:
    """
    Calculate the gravity metric for a 2 year time period.

    Args:
    start_year (int): The year at the start of the period.

    Returns:
    pd.Series: A series of gravity values between stocks in the S&P 500 across time.
    """
    # Define end year
    end_year = start_year+1

    # Read in mass data and set index
    mass = pd.read_csv(f'../data/S&P500/market_caps/market_caps_{start_year}-01-01_2021-12-31.csv')
    mass = mass.reset_index()
    mass['date'] = pd.to_datetime(mass['date']).dt.tz_localize(None)
    mass = mass.set_index(['date','ticker']).sort_index()
    
    # Read distance data and set index
    distance = pd.read_csv(f'../data/S&P500/distances/distance_{start_year}-01-01_{end_year}-12-31.csv')
    distance['date'] = pd.to_datetime(distance['date'])
    distance['date'] = (
        pd.to_datetime(distance['date'])
        .dt.tz_localize(None)
    )
    distance = distance.set_index(
        ['date', 'stock_i', 'stock_j']
    ).sort_index()

    # Subset mass data to have the same date range as distance data
    # Factor model ends up cutting out about 2 years w/ momentum calculation
    # And factor model window size
    dates = distance.index.get_level_values('date').unique()
    mass = mass.loc[
        mass.index.get_level_values('date').isin(dates)
    ]

    # Convert mass df to multiindex that matches distance data
    idx = distance.index
    distance['mass_i'] = (
        mass.reindex(
            pd.MultiIndex.from_arrays(
                [
                    idx.get_level_values('date'),
                    idx.get_level_values('stock_i')
                ]
            )
        )['market_cap']
        .to_numpy()
    )

    distance['mass_j'] = (
        mass.reindex(
            pd.MultiIndex.from_arrays(
                [
                    idx.get_level_values('date'),
                    idx.get_level_values('stock_j')
                ]
            )
        )['market_cap']
        .to_numpy()
    )

    # Calculate mass product as needed
    distance['mass_product'] = (
        distance['mass_i']
        * distance['mass_j']
    )

    # Gravity calculation
    return distance['mass_product'] / distance['Distance']