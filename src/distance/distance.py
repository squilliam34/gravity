import numpy as np
import pandas as pd
from data.data_loader import load_prices, load_factor_data

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
    
    return pd.DataFrame(sig, index=vix.index, columns=['lambda'])