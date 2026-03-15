from data.data_loader import load_merged_data
import statsmodels.api as sm
import pandas as pd

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
        valid_tickers, merged_data = load_merged_data(tickers)
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

            y = window_data[(ticker, 'Returns')]

            X = window_data[[(ticker, 'Momentum')]].copy()

            X['Market Return'] = window_data['Market Return']
            X['Rate Change'] = window_data['Rate Change']

            model = sm.OLS(y, sm.add_constant(X)).fit()

            ticker_momentum_key = (ticker, 'Momentum')

            results.append({
                'date': date,
                'ticker': ticker,
                'beta_market': model.params['Market Return'],
                'beta_rate': model.params['Rate Change'],
                'beta_momentum': model.params[ticker_momentum_key]
            })

    return pd.DataFrame(results)