from data.data_loader import load_merged_data
import statsmodels.api as sm
import pandas as pd

def get_data(FILEPATH: str):
    """
    Load the list of ticker symbols from a CSV file.

    Parameters:
    - FILEPATH (str): The file path to the CSV file containing the ticker symbols.

    Returns:
    """
    # Assume csv file has a column named 'Ticker' with the list of ticker symbols
    try:
        tickers = pd.read_excel(FILEPATH)['Ticker'].tolist()
        return load_merged_data(tickers)
    except Exception as e:
        print(f"Error occurred while loading data: {e}")
        return None