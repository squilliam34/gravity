import yfinance as yf

def get_yf_summary(ticker: str) -> str:
    """
    Retrieves a summary describing the company's operations.

    Parameters
    - ticker (str): The stock ticker symbol (e.g., NVDA)

    Returns
    - str: A summary of the company.
    """
    company = yf.Ticker(ticker)
    return company.info['longBusinessSummary']