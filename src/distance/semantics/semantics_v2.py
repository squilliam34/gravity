import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import warnings
from bs4.builder import XMLParsedAsHTMLWarning
from tqdm import tqdm

# Ignore the XML parsed as HTML warning specifically
warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

HEADERS = {
    'User-Agent': 'William Fan wdfan0128@gmail.com',
}

def build_cik_dict(FILEPATH: str):
    """
    Build a dictionary of ciks and tickers.

    Args:
    FILEPATH (str): The path to an excel spreadsheet of tickers.

    Returns:
    dict: A dictionary of tickers and their cik codes.
    """
    # Retrieve ciks from sec
    url = 'https://www.sec.gov/files/company_tickers.json'
    response = requests.get(url, headers=headers)
    data = response.json()

    # Convert dictionary values into dataframe
    companies = pd.DataFrame.from_dict(
        data,
        orient='index'
    )

    # Get list of S&P 500 companies
    sp500 = pd.read_excel(FILEPATH)
    sp500.drop(columns=['Sector', 'Name'], inplace=True)
    
    # Fill ciks with leading 0s
    companies['cik'] = companies['cik_str'].astype(str).str.zfill(10)
    companies = companies[
        ['ticker', 'cik', 'title']
    ]

    tickers = [ticker for ticker in sp500['Ticker']]
    valid = []
    ciks = []
    for ticker in tickers:
        entry = companies.loc[companies['ticker'] == ticker]
        if entry.empty:
            continue
        cik = entry.iloc[0]['cik']
        ciks.append(cik)
        valid.append(ticker)

    return dict(zip(valid, ciks))