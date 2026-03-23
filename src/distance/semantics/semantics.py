import yfinance as yf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np

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