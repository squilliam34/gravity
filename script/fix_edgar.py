'''Script to fix 10-k issues'''
# Some companies had TOC false positives or Item1's that were covered in boilerplate, page numbers, links, etc
# R1: Of the 122 tickers I wanted to replace, only 4 failed, which I filled in manually
# R2: Had to run it on 7 more tickers that were getting clustered weirdly (7/7 success)
# R3: 10 more clustering weirdly (8/10)
from edgar import *
import pandas as pd
import os
import re
from tqdm import tqdm
from dotenv import load_dotenv
import sys

# Import cik retrieval function
from src.distance.semantics.semantics_v2 import build_cik_dict, retrieve_item1_batch

if len(sys.argv) < 2:
    print('Error: Please provide a year.')
    sys.exit(1)

VALID_YEARS = {'2010', '2015', '2020', '2025'}

YEAR = sys.argv[1]

if YEAR not in VALID_YEARS:
    print(f'Error: YEAR must be one of {sorted(VALID_YEARS)}')
    sys.exit(1)

CACHE_PATH = f'./data/cache/descriptions/{YEAR}/item1_cache.parquet'

load_dotenv()
EMAIL = os.getenv('EMAIL')

set_identity(EMAIL)

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


def extract_item1_edgar(ticker: str, cik:str, year: str=YEAR):
    """
    Pull latest 10-K and extract Item 1 Business.
    Uses edgartools parsed business section first,
    then falls back to section lookup.
    """
    try:
        i = 2026 - int(year)
        filing = Company(cik).get_filings(form='10-K')[i]

        # Ensure that the year is correct
        if filing.filing_date.year > int(year):
            return ''

        if filing is None:
            return ''
        tenk = filing.obj()
        if tenk is None:
            return ''

        # Preferred method: parsed Business section
        try:
            business = tenk.business

            if business:
                text = clean_text(str(business))

                if len(text.split()) > 100:
                    return text

        except Exception:
            pass

        # Fallback: explicit Item 1 section
        try:
            item1 = tenk.sections['part_i_item_1']

            if item1:
                text = clean_text(item1.text())

                if len(text.split()) > 100:
                    return text

        except Exception:
            pass

        # Last fallback: scan sections
        try:
            for name, section in tenk.sections.items():
                if 'business' in name.lower():
                    text = clean_text(section.text())

                    if len(text.split()) > 100:
                        return text

        except Exception:
            pass

        return ''
    except Exception as e:
        print(f'{ticker} failed: {e}')
        return ''

def repair_item1_cache(cik_dict:dict[str, str], year: str = YEAR):

    # Load existing cache
    if os.path.exists(CACHE_PATH):
        cache = pd.read_parquet(CACHE_PATH)

        # Ensure expected columns exist
        if 'ticker' not in cache.columns:
            cache = cache.reset_index()

    else:
        cache = pd.DataFrame(columns=['ticker', 'item1_text'])

    # Use ticker as index for fast updates
    cache = cache.set_index('ticker')

    errors = []
    processed = 0

    for ticker, cik in tqdm(cik_dict.items()):

        # Skip tickers that already have a repaired Item 1
        if ticker in cache.index:
            existing = cache.loc[ticker, 'item1_text']

            if isinstance(existing, str) and existing.strip():
                continue
        
        item1 = extract_item1_edgar(ticker=ticker, cik=cik, year=year)

        if item1 == '':
            errors.append(ticker)
            print(f'{ticker} failed.')

        # Update cache in-place
        cache.loc[ticker] = item1

        processed += 1

        # Checkpoint every 25 processed companies
        if processed % 25 == 0:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            cache.reset_index().to_parquet(CACHE_PATH, index=False)
            print(f'Checkpoint saved ({processed} companies processed).')

    # Final save
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache.reset_index().to_parquet(CACHE_PATH, index=False)

    print(f'Processed {processed} companies.')
    print(f'Failed {len(errors)} tickers: {errors}')

    return errors

tickers_df = pd.read_csv(f'./data/csv/{YEAR}/tickers.csv')
tickers = tickers_df['Ticker'].to_list()
cik_dict = build_cik_dict(tickers)

print('Retrieving Item 1\'s using EdgarTools.')
errors = repair_item1_cache(cik_dict=cik_dict, year=YEAR)

# Try retrieving failed tickers using regex
failed_dict = {
    ticker: cik
    for ticker, cik in cik_dict.items()
    if ticker in errors
}

print('Retrieving Item 1\'s using RegEx.')
regex_results = retrieve_item1_batch(failed_dict, year=YEAR)