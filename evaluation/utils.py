import pandas as pd
import yfinance as yf
from src.cluster import Cluster
from config import DATA_DIR
from src.portfolio.portfolio import Portfolio, HoldingPeriod
from src.portfolio.portfolio_construction import *

def create_end_date(year:int):
    # Cap the end dates at the last day of data
    LAST_DATA_DATE = pd.Timestamp('2026-07-17')
    end_date = min(
        pd.Timestamp(f'{year+4}-12-31'),
        LAST_DATA_DATE
    )

    return end_date

def extract_leaders(
    year:int, 
    freq:str
) -> pd.DataFrame:

    end_date = create_end_date(year)

    periods = create_rebalance_periods(
        start_date=f'{year}-01-01',
        end_date=end_date,
        freq=freq
    )

    cluster_df = pd.read_csv(
        DATA_DIR 
        / 'clusters' 
        / f'{year}' 
        / 'clusters.csv'
    )
    clusters = []
    for label, group in cluster_df.groupby('cluster'):
        clusters.append(Cluster(label=label, tickers=group['ticker'].tolist()))

    leaders = get_period_leaders(periods=periods, clusters=clusters)

    return leaders

def construct_portfolio(
    year:int,
    freq:str
):
    leaders = extract_leaders(year=year, freq=freq)
    period_end = create_end_date(year)
    dates = sorted(leaders['date'].unique())

    holding_periods = []

    for i, start in enumerate(dates):

        if i == len(dates) - 1:
            end = period_end  # whatever your backtest end is
        else:
            end = dates[i + 1]

        period_df = leaders[
            (leaders['date'] == start)
            & (leaders['status'] == 'success')
        ].copy()

        tickers = period_df['leader'].tolist()

        if not tickers: continue

        weights = weight_portfolio(
            tickers,
            schema='equal'
        )

        holding_periods.append(
            HoldingPeriod(
                period=(start, end),
                holdings=weights
            )
        )

    return Portfolio(holding_periods)

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

    if tnx.empty:
        raise ValueError('No ^TNX data returned.')

    return (tnx['Close'] / 100).mean().item()
     