from data.data_loader import load_factor_data, get_tickers
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

def get_data(FILEPATH: str) -> pd.DataFrame:
    """
    Load the list of ticker symbols from a CSV file.

    Args:
    FILEPATH (str): The file path to the CSV file containing the ticker symbols.

    Returns:
    pd.DataFrame: The DataFrame containing stock price data, 10-Year Treasury data, and 
      S&P returns
    """
    # Assume csv file has a column named 'Ticker' with the list of ticker symbols
    try:
        tickers = get_tickers(FILEPATH)
        return load_factor_data(tickers)
    except Exception as e:
        print(f"Error occurred while loading data: {e}")
        return pd.DataFrame()
    
def calculate_rolling_betas(data: pd.DataFrame, 
                            tickers: list[str],
                            window: int = 252, 
                            ) -> pd.DataFrame:
    """
    Calculate rolling betas for each stock in the list of tickers using 
    a factor model that includes market return, rate change, and momentum.

    Args:
    data (pd.DataFrame): The merged DataFrame containing stock returns, 
      market returns, rate changes, and momentum.
    tickers (list[str]): The list of tickers to calculate the rolling betas for.
    window (int): The size of the rolling window (default is 252 trading days, 
      approximately one year).

    Returns:
    pd.DataFrame: A DataFrame containing the rolling betas for each stock and factor over time.
    """
    print('Running factor model...')
    # Create the month period directly from the index without adding it as a column
    months_series = data.index.to_period('M')
    month_ends = months_series.drop_duplicates()

    results = []

    market = data['Market Return'].values
    momentum = data['Momentum'].values
    rate = data['Rate Change'].values
    returns = data[tickers].values

    # Build a lookup dictionary for indices 
    month_indices = {}
    for idx, m in enumerate(months_series):
        month_indices.setdefault(m, []).append(idx)

    # Declare indexer
    t = 0
    for date in month_ends:

        # Get number of trading days using our fast lookup
        increment = len(month_indices[date])

        # Ensure that t > window:
        if t > window:

            Y = returns[t-window:t]
            MOM = momentum[t-window:t]
            MKT = market[t-window:t]
            RATE = rate[t-window:t]

            for i, ticker in enumerate(tickers):

                y_i = Y[:, i]

                mask = (
                    ~np.isnan(y_i) &
                    ~np.isnan(MOM) &
                    ~np.isnan(MKT) &
                    ~np.isnan(RATE)
                )

                if np.sum(mask) < 150:
                    continue

                X_i = np.column_stack([
                    np.ones(np.sum(mask)),
                    MKT[mask],
                    RATE[mask],
                    MOM[mask]
                ])

                y_clean = y_i[mask]

                XtX = X_i.T @ X_i
                XtY = X_i.T @ y_clean

                beta = np.linalg.solve(
                    XtX + 1e-8 * np.eye(XtX.shape[0]),
                    XtY
                )

                results.append({
                    'date': date, 
                    'ticker': ticker,
                    'beta_market': beta[1],
                    'beta_rate': beta[2],
                    'beta_momentum': beta[3]
                })
        
        t += increment

    df = pd.DataFrame(results)
    df.set_index(['date', 'ticker'], inplace=True)
    return df

def mahalanobis_distance(snapshot: pd.DataFrame, 
                         features: list[str] = [
                             'beta_market', 
                             'beta_rate', 
                             'beta_momentum'
                             ]) -> pd.DataFrame:
    """
    Calculate the Mahalanobis Distance between stocks at a given window in time for the given features.

    The Mahalanobis Distance is a multi-dimensional measure of the distance between a point and a distribution. 
    Unlike Euclidean distance, which treats all variables equally and assumes they are independent, Mahalanobis 
    distance accounts for the correlations between variables and is scale-invariant.

    Args:
    snapshot (pd.DataFrame): The DataFrame containing the stocks and their factor attributions at a point in time.
    features (list[str]): A list of features to use in order to calculate the Mahalanobis Distance.

    Returns:
    pd.DataFrame: A DataFrame containing the Mahalanobis Distance between stocks for the designated period.
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

def compute_distances(betas: pd.DataFrame, 
                      features: list[str] = [
                          'beta_market',
                          'beta_rate',
                          'beta_momentum'
                          ]) -> pd.DataFrame:
    """
    Calculate the Mahalanobis Distances for each point in time across all stocks available at that point.

    Args:
    betas (DataFrame): The DataFrame containing the list of companies, periods, and betas to compute
      the distance for.
    features (list[str]): A list of the features to use to calculate the distance for.

    Returns:
    pd.DataFrame: A DataFrame containing the distances between each stock for each window.
    """

    print('Calculating distances between betas...')
    results = []
    
    # Ensure 'date' and 'ticker' exist in columns
    if 'date' not in betas.columns or 'ticker' not in betas.columns:
        betas = betas.reset_index()
    for date, snapshot in betas.groupby('date'):
        # Transform distances to scale between 0-1 to align with cosine distance
        # Also helps to compress larger distances that lose economic meaning
        vec = mahalanobis_distance(snapshot, features)
        distances = 1 - np.exp(-vec)
        tickers = snapshot['ticker'].values

        # Get indices of upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(len(tickers), k=1)
        results.append(pd.DataFrame({
            'month': date,
            'stock_i': tickers[triu_idx[0]],
            'stock_j': tickers[triu_idx[1]],
            'factor distance': distances[triu_idx]
        }))

    results = pd.concat(results, ignore_index=True)

    # Convert to multindex 
    return results.set_index(
        ['month', 'stock_i', 'stock_j']
        ).sort_index()