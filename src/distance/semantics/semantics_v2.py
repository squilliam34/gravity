import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import warnings
from bs4.builder import XMLParsedAsHTMLWarning
from tqdm import tqdm
from dotenv import load_dotenv
import os
import numpy as np
from google import genai

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

GEMINI = genai.Client(
    api_key=os.getenv('GEMINI_KEY'))

def build_cik_dict(tickers: list[str], headers: dict=HEADERS):
    """
    Build a dictionary of ciks and tickers.

    Args:
    tickers (list[str]): The list of tickers to build a cik_dict for.

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
    
    # Fill ciks with leading 0s
    companies['cik'] = companies['cik_str'].astype(str).str.zfill(10)
    companies = companies[
        ['ticker', 'cik', 'title']
    ]

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

def get_tenk(cik: str, headers: dict = HEADERS) -> str:
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

def html_to_text(html: str) -> str:
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

def find_item1_start(text: str) -> int:
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
        candidates.extend(
            m.start() for m in re.finditer(pattern, text, re.I)
        )

    if not candidates:
        return None

    scored_candidates = []

    for pos in candidates:

        # Look after heading
        section_preview = text[pos:pos+5000]

        score = 0

        # Need substantial text
        words = section_preview.split()
        if len(words) > 200:
            score += 2

        # Penalize table-of-content behavior
        toc_indicators = [
            r'item\s+1a',
            r'item\s+2',
            r'item\s+3',
            r'item\s+4',
            r'page',
            r'\.{3,}',       # dotted TOC leaders
        ]

        toc_hits = sum(
            len(re.findall(pattern, section_preview, re.I))
            for pattern in toc_indicators
        )
        score -= toc_hits * 2

        # Reward prose-like content
        # Real sections have lots of sentence punctuation
        sentences = len(
            re.findall(r'[.!?]\s+[A-Z]', section_preview)
        )
        if sentences > 10:
            score += 3

        # Reward paragraph length
        avg_word_len = np.mean(
            [len(w) for w in words[:200]]
        )
        if avg_word_len > 4:
            score += 1
        scored_candidates.append(
            (score, pos)
        )

    # choose highest scoring candidate
    scored_candidates.sort(
        reverse=True
    )
    return scored_candidates[0][1]

def extract_item1(html_text, max_words=MAX_WORDS) -> str:
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

def normalize(array: np.ndarray) -> np.ndarray:
    """
    Normalize a vector using L2 normalization so that its norm equals 1.

    This projects the vector onto the unit sphere, ensuring that comparisons 
    focus on direction rather than magnitude. This is particularly useful when 
    using cosine similarity, where only the angle between vectors matters.

    Args:
    array (np.ndarray): 1D array (vector) to normalize.

    Returns:
    np.ndarray: L2-normalized vector.
    """
    # If it's a 1D vector, axis=0. If it's a 2D matrix, axis=1.
    axis = 1 if array.ndim > 1 else 0

    keepdims=True if array.ndim > 1 else False
    
    return array / np.linalg.norm(array, axis=axis, keepdims=keepdims)

def embed_text(descriptions: pd.DataFrame) -> np.ndarray:
    """
    Embed the texts from the DataFrame of descriptions generated by
    `retrieve_item1_batch`.
    
    Args:
    descriptions (pd.DataFrame): The DataFrame of descriptions to embed.

    Returns:
    tuple: First item is the embeddings matrix. Second is the number of descriptions 
      that got dropped.
    """
    # Sometimes the Item 1 comes back blank
    embed_df = descriptions.copy()

    embed_df = embed_df[
        embed_df['item1_text']
            .fillna('')
            .str.strip()
            .ne('')
    ]
    text = embed_df['item1_text'].tolist()

    # Batch embeddings
    response = GEMINI.models.embed_content(
        model='gemini-embedding-001',
        contents=text,
        config=genai.types.EmbedContentConfig(task_type='SEMANTIC_SIMILARITY')
    )
    embeddings = [
        e.values 
        for e in response.embeddings
    ]

    # Convert to numpy
    X = np.vstack(embeddings)
    return normalize(X), len(descriptions) - len(embed_df)

def calculate_cosine_similarity_distance(matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the distance in semantic meaning between stocks in a matrix using
    the cosine similarity. Initially calculates the cosine similarity between every
    element within the matrix then subtracts from 1 to arrive at a distance value.

    Parameters:
    - matrix (np.ndarray): The matrix of embeddings to calculate distances between.

    Returns:
    - np.ndarray: A matrix of distance values between every company in the matrix.
    """
    dist = 1 - np.clip(matrix @ matrix.T, -1, 1)
    np.fill_diagonal(dist, 0.0)
    return dist

def get_semantic_distances(tickers: list[str]) -> pd.DataFrame:
    """
    Takes list tickers and calculates the differences in semantic meaning
    between them. It embeds a description of each company's operations and calculates a 
    distance measure using the cosine similarity between each stock.

    Args:
    tickers (list[str]): A list of tickers to get the distances between between.

    Returns:
    pd.DataFrame: A DataFrame that represents a matrix of distances, indexed by ticker symbols.
    """

    # Retrieve descriptions
    print('Retrieving company descriptions...')
    ciks = build_cik_dict(tickers)
    descriptions = retrieve_item1_batch(ciks)

    # Embed descriptions
    print('Embedding descriptions...')
    result = embed_text(descriptions)
    embeddings = result[0]
    print(f'Dropped companies = {result[1]}')
    valid = result[2]

    print('Calculating distances...')
    # Calculate distances between each company using cosine similarity (1-cos(theta))
    matrix = calculate_cosine_similarity_distance(embeddings)

    df = pd.DataFrame(matrix, index=valid, columns=valid)

    # Mask upper triangle (exclude diagonal + duplicates)
    mask = np.triu(np.ones(df.shape), k=1).astype(bool)

    long_df = (
        df.where(mask)
          .stack()
          .to_frame('distance')
    )

    long_df.index.names = ['stock_i', 'stock_j']

    return long_df