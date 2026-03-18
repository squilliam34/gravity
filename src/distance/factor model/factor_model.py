from data.data_loader import load_factor_data
import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

def get_data(FILEPATH: str) -> tuple[list[str], pd.DataFrame]:
    """
    Load the list of ticker symbols from a CSV file.

    Parameters:
    - FILEPATH (str): The file path to the CSV file containing the ticker symbols.

    Returns:
    - Tuple[list[str], pd.DataFrame]: A tuple containing the list of valid ticker symbols 
    and the merged data DataFrame.
    """
    # Assume csv file has a column named 'Ticker' with the list of ticker symbols
    try:
        tickers = pd.read_excel(FILEPATH)['Ticker'].tolist()
        valid_tickers, merged_data = load_factor_data(tickers)
        return valid_tickers, merged_data
    except Exception as e:
        print(f"Error occurred while loading data: {e}")
        return None, pd.DataFrame()
    
def calculate_rolling_betas(data: pd.DataFrame, 
                            tickers: list, 
                            window: int = 252, 
                            step: int = 42
                            ) -> pd.DataFrame:
    """
    Calculate rolling betas for each stock in the list of tickers using 
    a factor model that includes market return, rate change, and momentum.

    Parameters:
    - data (DataFrame): The merged DataFrame containing stock returns, 
    market returns, rate changes, and momentum.
    - tickers (list): A list of ticker symbols for which to calculate the rolling betas.
    - window (int): The size of the rolling window (default is 252 trading days, 
    approximately one year).
    - step (int): The step size for rolling the window (default is 42 trading days, 
    approximately two months).

    Returns:
    - DataFrame: A DataFrame containing the rolling betas for each stock and factor over time.
    """
    results = []

    for t in range(window, len(data), step):

        window_data = data.iloc[t-window:t]
        date = data.index[t]

        for ticker in tickers:
            ticker_momentum_key = (ticker, 'Momentum')
            y = window_data[(ticker, 'Returns')]

            X = window_data[[ticker_momentum_key, 'Market Return', 'Rate Change']]
            valid = pd.concat([y, X], axis=1).dropna()
            # If the number of entries available is less than the threshold, continue onto next
            if len(valid) < 150:
                continue   

            y_valid = valid[(ticker, 'Returns')]
            X_valid = valid[[ticker_momentum_key, 'Market Return', 'Rate Change']]

            model = sm.OLS(y_valid, sm.add_constant(X_valid)).fit()
            results.append({
                'date': date,
                'ticker': ticker,
                'beta_market': model.params['Market Return'],
                'beta_rate': model.params['Rate Change'],
                'beta_momentum': model.params[ticker_momentum_key]
            })

    return pd.DataFrame(results)

def mahalanobis_distance(snapshot: pd.DataFrame, features: list[str] = ['beta_market', 
                                                                        'beta_rate', 
                                                                        'beta_momentum']):
    """
    Calculate the Mahalanobis Distance between stocks at a given window in time for the given features.

    The Mahalanobis Distance is a multi-dimensional measure of the distance between a point and a distribution. 
    Unlike Euclidean distance, which treats all variables equally and assumes they are independent, Mahalanobis 
    distance accounts for the correlations between variables and is scale-invariant.

    Parameters:
    - snapshot (DataFrame): The DataFrame containing the stocks and their factor attributions at a point in time.
    - features: (list[str]): A list of features to use in order to calculate the Mahalanobis Distance.

    Returns:
    - DataFrame: A DataFrame containing the Mahalanobis Distance between stocks for the designated period.
    """
    X = snapshot[features].values
    cov = np.cov(X, rowvar=False)
    # Add small regularization in case cov is singular
    # Adding a tiny amount of variance to each factor
    # And removing perfect multicollinearity
    cov += np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(cov)
    dist_matrix = squareform(pdist(X, metric='mahalanobis', VI=inv_cov))
    return dist_matrix

def compute_distances(betas, features = ['beta_market', 'beta_rate', 'beta_momentum']):
    """
    Calculate the Mahalanobis Distances for each point in time across all stocks available at that point.

    Parameters:
    - betas (DataFrame): The DataFrame containing the list of companies, periods, and betas to compute
    the distance for.
    - features (list[str]): A list of the features to use to calculate the distance for.

    Returns:
    - DataFrame: A DataFrame containing the distances between each stock for each window.
    """
    results = []

    for date, snapshot in betas.groupby("date"):

        distances = mahalanobis_distance(snapshot, features)

        tickers = snapshot["ticker"].values

        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                results.append({
                    "date": date,
                    "stock_i": tickers[i],
                    "stock_j": tickers[j],
                    "distance": distances[i, j]
                })

    return pd.DataFrame(results)