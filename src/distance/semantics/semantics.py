import yfinance as yf
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from data.data_loader import get_tickers

def get_yf_summary(ticker: str) -> str:
    """
    Retrieves a summary describing the company's operations.

    Args:
    ticker (str): The stock ticker symbol (e.g., NVDA)

    Returns:
    str: A summary of the company.
    """
    company = yf.Ticker(ticker)
    return company.info['longBusinessSummary']

def clean_text(text: str):
    """
    Clean input text by removing punctuation and non-alphanumeric characters and
    converting to lower case.

    Args:
    text (str): The input text to clean.

    Returns:
    str: The cleaned text.
    """
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

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

def embed_text(descriptions: list[str], 
               model_name: str='sentence-transformers/all-mpnet-base-v2'
               ) -> np.ndarray:
    """
    Produces embeddings of business descriptions text in a single matrix. 

    Args:
    descriptions (list[str]): A list of company descriptions to embed.
    model_name (str): The name of the model to use to embed the text.

    Returns:
    np.ndarray: A matrix that contains embeddings of the company descriptions. 
      Should return shape of (n x 768), where n is the number of companies.
    """
    # Declare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the texts
    # Padding and truncation ensure the tensor shapes match up to BERT's 512-token limit
    inputs = tokenizer(descriptions, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Pass inputs through model to extract hidden states
    with torch.no_grad():
        model_output = model(**inputs)

    # Perform Mean Pooling to get a single 768-dimensional vector per description
    token_embeddings = model_output.last_hidden_state 
    # Use the attention mask to ignore padding tokens so they don't skew the average
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    embeddings = (sum_embeddings / sum_mask).numpy() 

    return normalize(embeddings)

def calculate_cosine_similarity_distance(matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the distance in semantic meaning between stocks in a matrix using
    the cosine similarity. Initially calculates the cosine similarity between every
    company's embeddings within the matrix then subtracts from 1 to arrive at a 
    distance value. 

    There is no need to divide by the L2 norm of the vectors since each embedding has 
    already been normalized to have norm = 1.

    Args:
    matrix (np.ndarray): The matrix of embeddings to calculate distances between.

    Returns:
    np.ndarray: A matrix of distance values between every company in the matrix.
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

    # Create array containing embeddings for each company
    matrix = np.array([embed_text(get_yf_summary(ticker)) for ticker in tickers])

    # Calculate distances between each company using cosine similarity (1-cos(theta))
    distance_matrix = calculate_cosine_similarity_distance(matrix)

    # Return DataFrame of distances indexed by ticker
    return pd.DataFrame(distance_matrix, index=tickers, columns=tickers)