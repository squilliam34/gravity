'''Script to fix 10-k issues'''
# Some companies had TOC false positives or Item1's that were covered in boilerplate, page numbers, links, etc
from edgar import *
import pandas as pd
import os
import re
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
EMAIL = os.getenv('EMAIL')

set_identity(EMAIL)

CACHE_PATH = './data/cache/descriptions/item1_cache.parquet'


def clean_text(text: str) -> str:
    """
    Remove obvious SEC formatting garbage.
    """
    if not isinstance(text,str):
        return ''

    # remove excessive whitespace
    text = re.sub(r'\s+',' ',text)

    # remove page numbers / weird artifacts
    text = re.sub(r'\b\d+\s*$','',text)

    # remove repeated dots (TOC artifacts)
    text = re.sub(r'\.{3,',' ',text)

    return text.strip()


def extract_item1_edgar(ticker: str):
    """
    Pull latest 10-K and extract Item 1.
    """

    try:
        filing = Company('AAPL').get_filings(form='10-K')[0]
        tenk = filing.obj()

        item1 = tenk.sections['part_i_item_1']
        if item1 is None:
            return ''

        return clean_text(item1.text())

    except Exception as e:
        print(f'{ticker} failed: {e}')
        return ''

def repair_item1_cache(tickers):

    if os.path.exists(CACHE_PATH):
        cache = pd.read_parquet(CACHE_PATH)
    else:
        cache = pd.DataFrame(columns=['ticker','item1'])

    updates = []
    errors = []

    for ticker in tqdm(tickers):
        item1 = extract_item1_edgar(ticker)
        if item1 == '':
            errors.append(ticker)
        else:
            updates.append(
                {'ticker': ticker, 'item1_text': item1}
            )

    updates = pd.DataFrame(updates)

    if not updates.empty:
        # Replace only repaired tickers
        cache = cache[~cache.ticker.isin(tickers)]
        cache = pd.concat([cache, updates], ignore_index=True)
        cache.to_parquet(CACHE_PATH, index=False)

    print(f'Updated {len(updates)} tickers.')
    print(f'Failed {len(errors)} tickers: {errors}')


bad_tickers = ['AFL', 'ALL', 'COHR', 'CBRE', 'IDXX', 'EME', 'HSIC', 'ICE', 'GOOG', 'GOOGL', 
'GNRC', 'HST', 'DHR', 'CSGP', 'LHX', 'MO', 'MU', 'STLD', 'STT', 'TXT', 'VLTO', 'WM', 'TER', 
'PKG', 'PNC', 'NDAQ', 'PFE', 'REG', 'QCOM', 'DAL', 'ACN', 'BF-B', 'NI', 'KDP', 'TAP', 'STZ', 
'MNST', 'JNJ', 'INCY', 'MA', 'UHS', 'TSLA', 'TMO', 'TT', 'KO', 'MDT', 'NWSA', 'NWS', 'IBKR', 
'BAX', 'F', 'EFX', 'GM', 'CTVA', 'ALLE', 'ADP', 'APO', 'CMCSA', 'COR', 'BMY', 'BDX', 'AIG', 
'AXON', 'APTV', 'PTC', 'MCD', 'KIM', 'INTC', 'REGN', 'SYF', 'TKO', 'VRTX', 'EXPD', 'BRO', 'AMCR', 
'AMD', 'GE', 'DE', 'COST', 'MSFT', 'CIEN', 'CDW', 'CRWD', 'DELL', 'AKAM', 'BBY', 'ANET', 'AVGO', 
'SMCI', 'WDC', 'STX', 'FTNT', 'FFIV', 'IBM', 'HPQ', 'HPE', 'LITE', 'PANW', 'NVDA', 'SNDK', 'ORCL', 
'HAS', 'EL', 'DOW', 'BKR', 'ADM', 'AMGN', 'FDS', 'CRL', 'CME', 'COF', 'CMS', 'NEE', 'ITW', 'IT', 
'LYV', 'SYY', 'SLB', 'WMT']

repair_item1_cache(bad_tickers)