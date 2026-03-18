from data.data_loader import load_factor_data
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
                            window: int = 252, 
                            step: int = 42
                            ) -> pd.DataFrame:
    """
    Calculate rolling betas for each stock in the list of tickers using 
    a factor model that includes market return, rate change, and momentum.

    Parameters:
    - data (DataFrame): The merged DataFrame containing stock returns, 
    market returns, rate changes, and momentum.
    - window (int): The size of the rolling window (default is 252 trading days, 
    approximately one year).
    - step (int): The step size for rolling the window (default is 42 trading days, 
    approximately two months).

    Returns:
    - DataFrame: A DataFrame containing the rolling betas for each stock and factor over time.
    """
    # Get which columns are ticker dependent
    ticker_cols = [col for col in data.columns if isinstance(col, tuple)]
    ticker_data = data[ticker_cols]

    # Convert to MultiIndex if column is ticker dependent
    ticker_data.columns = pd.MultiIndex.from_tuples(ticker_data.columns)

    # Slice data so that tickers are now column names
    returns_df = ticker_data.xs('Returns', level=1, axis=1)
    momentum_df = ticker_data.xs('Momentum', level=1, axis=1)

    market = data['Market Return'].values
    rate = data['Rate Change'].values
    returns = returns_df.values        # (T, N)
    momentum = momentum_df.values      # (T, N)
    tickers = returns_df.columns.tolist()

    results = []

    for t in range(window, len(data), step):

        # Get the date and ensure proper format
        date = data.index[t].strftime('%Y-%m-%d')

        Y = returns[t-window:t, :]
        MOM = momentum[t-window:t, :]
        MKT = market[t-window:t]
        RATE = rate[t-window:t]

        for i, ticker in enumerate(tickers):

            y_i = Y[:, i]
            mom_i = MOM[:, i]

            # Checks how many entries are totally NaN
            mask = (
                ~np.isnan(y_i) &
                ~np.isnan(mom_i) &
                ~np.isnan(MKT) &
                ~np.isnan(RATE)
            )

            # If a window has less than 150 valid entries, continue on
            # Check to ensure we have enough data for the ticker in the window
            if np.sum(mask) < 150:
                continue

            # Stack columns into a matric
            X_i = np.column_stack([
                np.ones(np.sum(mask)),
                MKT[mask],
                RATE[mask],
                mom_i[mask]
            ])

            y_clean = y_i[mask]

            # OLS solution for beta = (X^TX)^-1 (X^Ty)
            XtX = X_i.T @ X_i
            XtY = X_i.T @ y_clean
            # Computes betas with small regularization if XtX is singular
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
    df = pd.DataFrame(results)
    df.set_index(['date', 'ticker'], inplace=True)

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