import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import warnings
from bs4.builder import XMLParsedAsHTMLWarning
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

# Ignore the XML parsed as HTML warning specifically
warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

HEADERS = {
    'User-Agent': os.getenv('SEC_USER_AGENT'),
}

# Limit on Item 1 length
MAX_WORDS = 4000

# Possible Item 1 headers
ITEM1_PATTERNS = [
    r'\bitem\s+1\s*[\.\-:]*\s*business\b',
    r'\bitem\s+1\s*[\.\-:]*\s*the\s+company\b',
    r'\bitem\s+1\s*[\.\-:]*\s*company\s+overview\b',
    r'\bitem\s+1\s*[\.\-:]*\s*overview\b',
    r'\bitem\s+1\s*[\.\-:]*\s*description\b',
]

ITEM1_END_PATTERN = re.compile(
    r'\bitem\s+1a\b'
    r'|\bitem\s+2\b'
    r'|\bitem\s+1b\b',
    re.IGNORECASE
)

def build_cik_dict(FILEPATH: str, headers: dict=HEADERS):
    '''
    Build a dictionary of ciks and tickers.

    Args:
    FILEPATH (str): The path to an excel spreadsheet of tickers.

    Returns:
    dict: A dictionary of tickers and their cik codes.
    '''
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
    sp500.drop(columns=[col for col in sp500.columns if col not in ['Ticker']], inplace=True)
    
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

def get_tenk(cik: str, headers: dict = HEADERS):
    '''
    Retrieve a company's 10-k from the SEC using its cik.

    Args:
    cik (str): The cik code of the desired company.
    headers (dict): The headers to use in order to access the SEC website.

    Returns:
    str: A string of the 10-k filing.
    '''
    submissions_url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    submissions = requests.get(
        submissions_url,
        headers=headers
    ).json()
    recent = pd.DataFrame(submissions['filings']['recent'])
    # Find the indexes of all 10-K filings

    tenk = recent[recent['form'] == '10-K']
    latest = tenk.iloc[0]
    accession = latest['accessionNumber'].replace('-', '')
    url = (
        f'https://www.sec.gov/Archives/edgar/data/'
        f'{int(cik)}/{accession}/{latest['primaryDocument']}'
    )

    html = requests.get(url, headers=headers).text
    return html

def html_to_text(html: str):
    '''
    Convert SEC filing HTML to clean text.

    Args:
    html (str): The html text of the 10-k from the SEC website.

    Returns:
    str: The cleaned text.
    '''

    soup = BeautifulSoup(html, 'lxml')

    # Remove non-content
    for tag in soup([
        'script',
        'style',
        'ix:header',
        'ix:hidden'
    ]):
        tag.decompose()

    text = soup.get_text(' ')

    # normalize whitespace
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def find_item1_start(text: str):
    '''
    Find candidate Item 1 locations. Avoid TOC false positives by 
    requiring sufficient content after the match.

    Args:
    text (str): The cleaned 10-k.

    Returns:
    int: A likely position of the start of Item 1.
    '''

    candidates = []

    for pattern in ITEM1_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            candidates.append(match.start())

    if not candidates:
        return None

    # Item 1 TOCs usually appear near beginning
    # Real Item 1 normally has lots of text after it.
    for pos in candidates:

        remaining = text[pos:pos+5000]

        # heuristic:
        # real section should contain words after heading
        if len(remaining.split()) > 100:
            return pos

    return candidates[0]

def extract_item1(html_text, max_words=MAX_WORDS):
    '''
    Extracts Item 1 from the 10-k.

    Args:
    html_text (str): The raw 10-k text.
    max_words (int): The maximum number of words to pull for Item 1.

    Returns:
    str: The Item 1 text, business overview.
    '''

    text = html_to_text(html_text)

    start = find_item1_start(text)
    if start is None:
        return ''

    # Start after heading itself
    section = text[start:]

    # Find end
    end = ITEM1_END_PATTERN.search(
        section,
        pos=300)

    if end:
        section = section[:end.start()]

    # Limit runaway extraction
    words = section.split()

    if len(words) > max_words:
        section = ' '.join(words[:max_words])

    section = section.strip()

    # Validation
    if len(section) < 500:
        return ''

    return section