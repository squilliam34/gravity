import yfinance as yf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np

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

def chunk_text(text: str,
               model_name: str='gpt-4',
               chunk_size: int=512,
               chunk_overlap: int=50):
    """
    Chunks the data as specified so as to not exceed token counts.

    Parameters
    - text (str): The text to chunk.
    - model_name (str): The name of the model encoding to use.
    - chunk_size (int): The number of tokens to allow in a chunk.
    - chunk_overlap (int): The number of tokens to save between chunks to save context.

    Returns
    - list[str]: a list containing the string processed into various chunks.
    """
    # Start with default recommendations of 512 word chunks + 10% overlap
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return splitter.split_text(text)