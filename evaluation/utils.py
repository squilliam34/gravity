import pandas as pd
import yfinance as yf
from functools import lru_cache
from src.cluster import Cluster
from config import DATA_DIR

def create_end_date(year:int):
    # Cap the end dates at the last day of data
    LAST_DATA_DATE = pd.Timestamp('2026-07-17')
    end_date = min(
        pd.Timestamp(f'{year+4}-12-31'),
        LAST_DATA_DATE
    )

    return end_date

@lru_cache(maxsize=None)
def get_risk_free_rate(
    start_date: str,
    end_date: str,
) -> float:
    """
    Return the average 10-year Treasury yield over the period as a decimal.

    Parameters
    ----------
    start_date : str
        YYYY-MM-DD
    end_date : str
        YYYY-MM-DD

    Returns
    -------
    float
        Average annual risk-free rate as a decimal
        (e.g. 0.0423 = 4.23%)
    """
    tnx = yf.download(
        '^TNX',
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if not tnx.empty:
        return (tnx['Close'] / 100).mean().item()

    # Yahoo occasionally returns no history for the ^TNX index. Fall back to
    # the Federal Reserve's equivalent daily 10-year Treasury series while
    # preserving this function's public interface.
    url = (
        'https://fred.stlouisfed.org/graph/fredgraph.csv'
        f'?id=DGS10&cosd={start_date}&coed={end_date}'
    )
    treasury = pd.read_csv(url)
    yields = pd.to_numeric(treasury['DGS10'], errors='coerce').dropna()

    if yields.empty:
        raise ValueError('No 10-year Treasury yield data returned.')

    return (yields / 100).mean()
