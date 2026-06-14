import numpy as np
import pandas as pd
from data.data_loader import load_prices, load_factor_data
from src.distance.factor_model.factor_model import compute_distances, calculate_rolling_betas
from src.distance.semantics.semantics import get_semantic_distances
from datetime import date

def sigmoid(vix: np.ndarray, k: int = 0.05, threshold: int=20) -> np.ndarray:
    """
    Apply the sigmoid transformation to the VIX to fit it between 0 and 1 as
    the lambda weight for my distance metric.

    Args:
    vix (np.ndarray): An array of closing VIX values.
    k (int): The tuning parameter. Determines how "steep" the sigmoid is.
    threshold (int): The long run average of the VIX (~20). Used to scale the current value.

    Returns:
    np.ndarray: An array containing the transformed VIX value at each point in time.
    """
    return 1/(1 + np.exp(-k*(vix-threshold)))

def get_lambda(start_date: str = '2000-01-01', 
                end_date: str = date.today().strftime('%Y-%m-%d'), 
                interval: str = '1d') -> pd.DataFrame:
    """
    Calculates lambda for the given time period by applying a sigmoid
    transformation to the VIX.

    Args:
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d' for daily, '1wk' for weekly).

    Returns:
    pd.DataFrame: A DataFrame containing the series of lambda values to use for 
      weighting in final distance calculation.
    """
    vix = load_prices('^VIX', start_date=start_date, end_date=end_date, interval=interval)
    vix.columns=['VIX']

    # Apply sigmoid transformation
    sig = sigmoid(vix['VIX'].to_numpy())
    
    sig_df = pd.DataFrame(sig, index=vix.index, columns=['lambda'])
    sig_df.index = pd.to_datetime(sig_df.index)
    
    return sig_df

def build_distances(tickers: list[str]) -> pd.DataFrame:
    """
    Calculate the final distance measure. Apply time-varying weights using the VIX
    to cosine distances (more structural measure) and factor distances (more behavioral).

    Args:
    tickers (list[str]): A list of tickers to get the distances between between.

    Returns:
    pd.DataFrame: A DataFrame containing the final distances for any stock combination across time
    """
    print('Loading factor data...')

    # Calculate distance of factors
    factor_data = load_factor_data(tickers)
    betas = calculate_rolling_betas(data=factor_data, tickers=tickers)
    beta_distance = compute_distances(betas=betas)

    # Create daily range of dates from the factor data since it has the latest start
    dates = beta_distance.index.get_level_values('month')
    start_date = dates.asfreq('D', how='start')[0].to_timestamp()
    today = pd.to_datetime(date.today().strftime('%Y-%m-%d'))
    daily_index = pd.date_range(start=start_date, end=today, freq='D')

    # Get lambda values and index to match factor time frame
    lam = get_lambda()
    lam = lam.loc[daily_index[0]: ]

    # Retrieve cosine differences
    print('Loading cosine differences...')
    semantic_distance = get_semantic_distances(tickers)

    # Create new DataFrame to house all the merged values
    pairs = pd.MultiIndex.from_product(
        [daily_index, tickers, tickers],
        names=['date', 'stock_i', 'stock_j']
    )

    # Ensures no duplicate pairings
    pairs = pairs[pairs.get_level_values('stock_i') < pairs.get_level_values('stock_j')]

    df = pd.DataFrame(index=pairs).reset_index()

    # Join time series
    print('Joining time series...')
    df = df.join(lam, on='date')
    df = df.join(semantic_distance, on=['stock_i','stock_j'])
    df['month'] = df['date'].dt.to_period('M')
    df = df.join(beta_distance, on=['month','stock_i','stock_j'])
    df.drop(columns='month', inplace=True)
    df.dropna(subset=['lambda', 'factor distance'], inplace=True)
    print('Time series joined successfully...')

    # Actually calculate lamba * D_f + (1-lambda) * D_c
    lambda_val = df['lambda']
    factors = df['factor distance']
    semantics = df['semantic distance']

    df['Distance'] = lambda_val*factors + (1 - lambda_val)*semantics

    df = df[['date', 'stock_i','stock_j', 'Distance']]
    df.set_index(['date', 'stock_i', 'stock_j'], inplace=True)
    df.sort_index(inplace=True)
    return df