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
import hashlib
from pathlib import Path
import time
from sklearn.decomposition import PCA

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

###################################
########## CACHE HELPERS ##########
###################################

EMBEDDING_CACHE = Path('./data/cache/embeddings/gemini_item1_raw_cache.parquet')

ITEM1_CACHE = Path('./data/cache/descriptions/item1_cache.parquet')

def text_hash(text: str) -> str:
    """
    Create unique identifier for a description.
    """
    return hashlib.md5(
        text.encode('utf-8')
    ).hexdigest()


def load_embedding_cache() -> pd.DataFrame:
    """
    Load existing embedding cache.
    """

    if EMBEDDING_CACHE.exists():
        return pd.read_parquet(EMBEDDING_CACHE)

    return pd.DataFrame(
        columns=[
            'ticker',
            'text_hash',
            'embedding'
        ]
    )

def save_embedding_cache(cache: pd.DataFrame):
    """
    Save embedding cache.
    """

    EMBEDDING_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cache.to_parquet(
        EMBEDDING_CACHE,
        index=False
    )

###################################
######### 10-k Functions ##########
###################################

def build_cik_dict(tickers: list[str], headers: dict=HEADERS):
    """
    Build a dictionary of ciks and tickers.

    Args:
    tickers (list[str]): The list of tickers to build a cik_dict for.
    headers (dict): The headers to use in order to access the SEC website.

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
    """
    Retrieve a company's 10-k from the SEC using its cik.

    Args:
    cik (str): The cik code of the desired company.
    headers (dict): The headers to use in order to access the SEC website.

    Returns:
    str: A string of the 10-k filing.
    """
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
    """
    Convert SEC filing HTML to clean text.

    Args:
    html (str): The html text of the 10-k from the SEC website.

    Returns:
    str: The cleaned text.
    """

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
    """
    Find candidate Item 1 locations. Avoid TOC false positives by 
    requiring sufficient content after the match.

    Args:
    text (str): The cleaned 10-k.

    Returns:
    int: A likely position of the start of Item 1.
    """

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
    """
    Extracts Item 1 from the 10-k.

    Args:
    html_text (str): The raw 10-k text.
    max_words (int): The maximum number of words to pull for Item 1.

    Returns:
    str: The Item 1 text, business overview.
    """

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

def retrieve_item1_batch(cik_dict: dict) -> pd.DataFrame:
    """
    Retrieve Item 1 business descriptions, using a local cache
    so descriptions are only extracted once.

    Args:
    cik_dict (dict): Dictionary mapping ticker -> CIK.

    Returns:
    pd.DataFrame containing ticker and Item 1 text.
    """

    # Load cache if it exists
    if ITEM1_CACHE.exists():
        cache = pd.read_parquet(ITEM1_CACHE)
    else:
        cache = pd.DataFrame(
            columns=['ticker', 'item1_text']
        )

    descriptions = {}

    for ticker, cik in tqdm(cik_dict.items(), desc='Extracting Item 1'):

        ticker = ticker.strip()

        # Check cache first
        cached = cache.loc[
            cache['ticker'] == ticker,
            'item1_text'
        ]

        if not cached.empty:
            descriptions[ticker] = cached.iloc[0]
            continue

        try:
            html = get_tenk(cik)
            item1 = extract_item1(html)

            descriptions[ticker] = item1

            # Immediately add to cache so progress isn't lost
            cache = pd.concat(
                [
                    cache,
                    pd.DataFrame({
                        'ticker': [ticker],
                        'item1_text': [item1]
                    })
                ],
                ignore_index=True
            )

        except Exception as e:
            descriptions[ticker] = None
            print(f'\nFailed {ticker}: {e}')

    # Save updated cache
    ITEM1_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cache.drop_duplicates(
        subset='ticker',
        keep='last'
    ).to_parquet(
        ITEM1_CACHE,
        index=False
    )

    return pd.DataFrame(descriptions.items(), columns=['ticker', 'item1_text'])

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
      that got dropped. Third is the kept tickers.
    """
    # Sometimes the Item 1 comes back blank
    embed_df = descriptions.copy()

    embed_df = embed_df[
        embed_df['item1_text']
        .fillna('')
        .str.strip()
        .ne('')
    ].copy()

    if embed_df.empty:
        return np.empty((0,3072)), len(descriptions), []

    # Generate hashes
    embed_df['text_hash'] = (
        embed_df['item1_text']
        .apply(text_hash)
    )

    cache = load_embedding_cache()

    # Split cached vs missing
    cached = embed_df.merge(
        cache,
        on=['ticker', 'text_hash'],
        how='inner'
    )

    missing = embed_df.merge(
        cache,
        on=['ticker', 'text_hash'],
        how='left',
        indicator=True
    )

    missing = missing[missing['_merge']=='left_only']

    print(f'Cached embeddings: {len(cached)}')
    print(f'New embeddings required: {len(missing)}')


    embeddings_dict = {}

    # Load cached embeddings
    for _, row in cached.iterrows():
        embeddings_dict[row['ticker']] = np.array(
            row['embedding']
        )

    # Embed new descriptions
    if len(missing) > 0:

        new_embeddings = []
        texts = missing['item1_text'].tolist()
        batch_size = 100

        for i in tqdm(
            range(0,len(texts),batch_size),
            desc='Embedding batches'
        ):

            batch = texts[i:i+batch_size]

            response = GEMINI.models.embed_content(
                model='gemini-embedding-001',
                contents=batch,
                config=genai.types.EmbedContentConfig(
                    task_type='SEMANTIC_SIMILARITY'
                )
            )

            new_embeddings.extend(
                [e.values for e in response.embeddings]
            )

            # Gemini API has 100k token/minute limit, which I hit after 2 batches
            time.sleep(30)

        new_cache = pd.DataFrame({
            'ticker': missing['ticker'].tolist(),
            'text_hash': missing['text_hash'].tolist(),
            'embedding': new_embeddings
        })

        cache = pd.concat([cache,new_cache], ignore_index=True)

        save_embedding_cache(cache)

        for ticker, emb in zip(missing['ticker'], new_embeddings):
            embeddings_dict[ticker] = np.array(emb)

    # Preserve ordering
    valid = embed_df['ticker'].tolist()

    X = np.vstack(
        [
            embeddings_dict[ticker]
            for ticker in valid
        ]
    )

    return (X, len(descriptions)-len(embed_df), valid)

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

    # Apply PCA to address curse of dimensionality
    pca = PCA(n_components=0.99)
    embeddings = pca.fit_transform(embeddings)

    # Normalize post PCA
    embeddings = normalize(embeddings)

    print('Calculating distances...')
    # Calculate distances between each company using cosine similarity (1-cos(theta))
    matrix = calculate_cosine_similarity_distance(embeddings)

    return pd.DataFrame(matrix, index=valid, columns=valid)