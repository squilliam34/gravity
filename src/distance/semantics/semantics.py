import yfinance as yf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from data.data_loader import get_tickers

def get_yf_summary(ticker: str) -> str:
    """
    Retrieves a summary describing the company's operations.

    Parameters:
    - ticker (str): The stock ticker symbol (e.g., NVDA)

    Returns:
    - str: A summary of the company.
    """
    company = yf.Ticker(ticker)
    return company.info['longBusinessSummary']

def chunk_text(text: str,
               model_name: str='gpt-4',
               chunk_size: int=512,
               chunk_overlap: int=50
               ) -> list[str]:
    """
    Chunks the data as specified so as to not exceed token counts.

    Parameters:
    - text (str): The text to chunk.
    - model_name (str): The name of the model encoding to use.
    - chunk_size (int): The number of tokens to allow in a chunk.
    - chunk_overlap (int): The number of tokens to save between chunks to save context.

    Returns:
    - list[str]: A list containing the string processed into various chunks.
    """
    # Start with default recommendations of 512 word chunks + 10% overlap
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return splitter.split_text(text)

def normalize(array: np.ndarray) -> np.ndarray:
    """
    Normalize a vector using L2 normalization so that its norm equals 1.

    This projects the vector onto the unit sphere, ensuring that comparisons 
    focus on direction rather than magnitude. This is particularly useful when 
    using cosine similarity, where only the angle between vectors matters.

    Parameters:
    - array (np.ndarray): 1D array (vector) to normalize.

    Returns:
    - np.ndarray: L2-normalized vector.
    """
    return array/np.linalg.norm(array)

def embed_text(text: str, 
               model_name: str='all-MiniLM-L6-v2') -> np.ndarray:
    """
    Produce a final embedding of chunked text in a single vector. 

    Parameters:
    - text (str): The string to process and generate embeddings for.
    - model_name (str): The name of the model to use to embed the text.

    Returns:
    - np.ndarray: A singular 1D vector that contains an embedding of the chunked text.
    """
    # Chunk text
    chunks = chunk_text(text)
    model = SentenceTransformer(model_name)
    embeddings = [model.encode(chunk) for chunk in chunks]
    
    # Normalize embeddings
    embeddings = [normalize(e) for e in embeddings]

    # Perform mean pooling to combine chunks into one vector
    embeddings = np.mean(embeddings, axis=0)

    # Normalize one more time and return
    return normalize(embeddings)

def calculate_cosine_similarity_distance(matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the distance in semantic meaning between stocks in a matrix using
    the cosine similarity. Initially calculates the cosine similarity between every
    company's embeddings within the matrix then subtracts from 1 to arrive at a 
    distance value. 

    There is no need to divide by the L2 norm of the vectors since each embedding has 
    already been normalized to have an L2 norm 1.

    Parameters:
    - matrix (np.ndarray): The matrix of embeddings to calculate distances between.

    Returns:
    - np.ndarray: A matrix of distance values between every company in the matrix.
    """
    dist = 1 - np.clip(matrix @ matrix.T, -1, 1)
    np.fill_diagonal(dist, 0.0)
    return dist

def get_semantic_distances(FILEPATH: str) -> pd.DataFrame:
    """
    Takes a file containing tickers and calculates the differences in semantic meaning
    between them. It embeds a description of each company's operations and calculates a 
    distance measure using the cosine similarity between each stock.

    Parameters:
    - FILEPATH (str): The path to the file containing the tickers to calculate distances
    between.

    Returns:
    - pd.DataFrame: A DataFrame that represents a matrix of distances, indexed by ticker symbols.
    """
    tickers = get_tickers(FILEPATH=FILEPATH)

    # Create array containing embeddings for each company
    matrix = np.array([embed_text(get_yf_summary(ticker)) for ticker in tickers])

    # Calculate distances between each company using cosine similarity (1-cos(theta))
    distance_matrix = calculate_cosine_similarity_distance(matrix)

    # Return DataFrame of distances indexed by ticker
    return pd.DataFrame(distance_matrix, index=tickers, columns=tickers)