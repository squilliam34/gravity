import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from src.cluster import Cluster
from evaluation.utils import create_end_date
from src.portfolio.portfolio import Portfolio, HoldingPeriod
from config import DATA_DIR
    
def get_valid_network_date(
    cluster: Cluster,
    start_date: str,
    end_date: str,
    threshold: float | None = None,
    force_recompute: bool = False
) -> tuple[pd.Timestamp | None, nx.Graph | None]:
    """
    Find the first date with an available network snapshot.

    Args:
    cluster (Cluster): The cluster for which to retrieve a network.
    start_date (str): The starting date for the search window.
    end_date (str): The ending date for the search window.
    threshold (float | None): Optional minimum edge weight required to include a connection.
    force_recompute (bool): If True, indicates to ignore cached data and recompute it.

    Returns:
    tuple[pd.Timestamp | None, networkx.Graph | None]: The first valid date and its network graph, or None values if no valid network is found.
    """

    date = pd.Timestamp(start_date)

    # Find the next valid trading day in the data
    for _ in range(5):

        G = cluster.get_network(
            start_date=date.strftime('%Y-%m-%d'), 
            end_date=end_date,
            threshold=threshold,
            force_recompute=force_recompute
        )

        if G is not None:
            return date, G

        date += pd.Timedelta(days=1)

    return None, None

def create_rebalance_periods(
    start_date: str,
    end_date: str,
    freq: str = '6MS'
) -> list[tuple[str, str]]:
    """
    Create semiannual rebalance periods.

    Args:
    start_date (str): Beginning of analysis period.
    end_date (str): End of analysis period.
    freq (str): Frequency for rebalancing. Default is every 6 months.

    Returns:
    list[tuple]: List of (rebalance_date, period_end) tuples.
    """
    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq=freq
    )

    periods = []

    for i, date in enumerate(dates):

        if i < len(dates) - 1:
            period_end = dates[i + 1] - pd.Timedelta(days=1)
        else:
            period_end = pd.Timestamp(end_date)

        periods.append((date.strftime('%Y-%m-%d'), period_end.strftime('%Y-%m-%d')))

    return periods

def get_period_leaders(
    periods: list[tuple[str, str]],
    clusters: list[Cluster],
    threshold: float | None = None,
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Identify the leader stock for each cluster during each rebalance period.

    For each rebalance period and each cluster, the function attempts to locate
    a valid network snapshot (via `get_valid_network_date`). If a network is
    found, it computes eigenvector centrality scores and selects the ticker
    with the highest centrality as the leader. The function records status
    codes for missing networks, too-small graphs, or centrality failures.

    Args:
    periods (list[tuple[str, str]]): Rebalance periods as `(start_date, end_date)` strings.
    clusters (list[Cluster]): List of `Cluster` instances to evaluate.
    threshold (float): Optional minimum edge weight required to include connections.
    force_recompute (bool): If True, force recomputation of any cached network data.

    Returns:
    pd.DataFrame: Rows with columns `date`, `cluster`, `leader`,
      `centrality`, and `status` describing the result for each cluster and
      rebalance period.
    """
    leaders = []
    for (period_start, period_end) in tqdm(
        periods,
        desc='Identifying leaders'
    ):
        for cluster in clusters:
            network_date, G = get_valid_network_date(
                cluster=cluster,
                start_date=period_start,
                end_date=period_end,
                threshold=threshold,
                force_recompute=force_recompute
            )

            # No network found
            if G is None:
                leaders.append({
                    'rebalance_date': pd.Timestamp(period_start),
                    'date': network_date,
                    'cluster': cluster.label,
                    'leader': None,
                    'centrality': None,
                    'status': 'no_network'
                })
                continue

            # Too few nodes / edges
            if len(G.nodes) < 2:
                leaders.append({
                    'rebalance_date': pd.Timestamp(period_start),
                    'date': network_date,
                    'cluster': cluster.label,
                    'leader': None,
                    'centrality': None,
                    'status': 'too_small'
                })
                continue

            try:
                centrality_scores = nx.eigenvector_centrality(G, weight='weight')

                leader, score = max(centrality_scores.items(), key=lambda x: x[1])

                leaders.append({
                    'rebalance_date': pd.Timestamp(period_start),
                    'date': network_date,
                    'cluster': cluster.label,
                    'leader': leader,
                    'centrality': score,
                    'status': 'success'
                })

            except nx.NetworkXException:

                leaders.append({
                    'rebalance_date': pd.Timestamp(period_start),
                    'date': network_date,
                    'cluster': cluster.label,
                    'leader': None,
                    'centrality': None,
                    'status': 'centrality_failed'
                })

    return pd.DataFrame(leaders)

def weight_portfolio(
    tickers: list[str],
    schema: str = 'equal',
    **kwargs: object
) -> dict[str, float]:
    """
    Return a mapping of ticker weights according to the requested schema.

    Supported schemas:
    - `equal`: assign equal weight to each ticker.
    - `cluster`: assign weight by the aggregate market cap of each leader's cluster.

    Args:
    tickers (list[str]): List of ticker symbols to weight.
    schema (str): Weighting schema name (default: 'equal').
    kwargs (dict[str, object]): Schema-specific inputs. The `cluster` schema
      requires a `market_caps` mapping keyed by leader ticker.

    Returns:
    dict: Mapping of ticker -> weight (floats summing to ~1.0).

    Raises:
    ValueError: If an unknown `schema` is provided.
    """

    if schema == 'equal':
        return equal_weights(tickers)

    if schema == 'cluster':
        market_caps = kwargs.get('market_caps')
        if market_caps is None:
            raise ValueError(
                'Cluster weighting requires a market_caps mapping.'
            )
        return cluster_weights(tickers, market_caps)

    raise ValueError(f'Unknown weighting schema: {schema}')

def equal_weights(
    tickers: list[str]
) -> dict[str, float]:
    """
    Assign equal weights to a list of tickers.

    Args:
    tickers (list[str]): Non-empty list of ticker symbols.

    Returns:
    dict[str, float]: Mapping of each ticker to its equal weight.

    Raises:
    ValueError: If `tickers` is empty.
    """

    if len(tickers) == 0:
        raise ValueError('Cannot weight an empty portfolio.')

    weight = 1 / len(tickers)

    return {ticker: weight for ticker in tickers}


def cluster_weights(
    tickers: list[str],
    market_caps: dict[str, float] | pd.Series
) -> dict[str, float]:
    """
    Weight each cluster leader by its cluster's aggregate market cap.
    
    Args:
    tickers (list[str]): Non-empty list of ticker symbols.
    market_caps (dict[str, float] | pd.Series): Market caps for each ticker.

    Returns:
    dict: A dictionary of weightings for each ticker
    """
    if len(tickers) == 0:
        raise ValueError('Cannot weight an empty portfolio.')

    caps = pd.Series(market_caps, dtype='float64').reindex(tickers)

    if caps.isna().any():
        missing = caps.index[caps.isna()].tolist()
        raise ValueError(f'Missing cluster market caps for: {missing}')
    if (~np.isfinite(caps) | (caps <= 0)).any():
        raise ValueError('Cluster market caps must be finite and positive.')

    weights = caps / caps.sum()
    return weights.to_dict()


def get_cluster_market_cap(
    cluster: Cluster,
    date: str | pd.Timestamp
) -> float:
    """
    Return the sum of constituent market caps at a cluster snapshot.
    
    Args:
    cluster (Cluster): The cluster to retrieve the aggregated market cap for.
    date (str | pd.Timestamp): The date in time to retrieve the aggregated market at.

    Returns:
    float: The aggregate market cap of the cluster.
    """
    snapshot_date = pd.Timestamp(date)
    market_caps = cluster.get_market_caps(
        start_date=snapshot_date.strftime('%Y-%m-%d'),
        end_date=snapshot_date.strftime('%Y-%m-%d')
    )

    if market_caps.empty:
        raise ValueError(
            f'No market-cap data for cluster {cluster.label} on '
            f'{snapshot_date:%Y-%m-%d}.'
        )

    # Market caps are stored as natural logs by create_market_cap_df.
    raw_market_caps = np.exp(market_caps['market_cap'])
    cluster_market_cap = raw_market_caps.sum(min_count=1)
    if not np.isfinite(cluster_market_cap) or cluster_market_cap <= 0:
        raise ValueError(
            f'Invalid market-cap data for cluster {cluster.label} on '
            f'{snapshot_date:%Y-%m-%d}.'
        )

    return float(cluster_market_cap)

def extract_leaders(
    year:int, 
    freq:str
) -> pd.DataFrame:
    """
    Extract cluster leaders for each rebalance period in a given year.

    Args:
    year (int): The year to analyze.
    freq (str): Rebalancing frequency used to build the periods.

    Returns:
    pd.DataFrame: Leader selection results for each cluster and period.
    """

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

    successful = leaders['status'].eq('success')
    cluster_lookup = {cluster.label: cluster for cluster in clusters}
    leaders.loc[successful, 'cluster_market_cap'] = leaders.loc[
        successful
    ].apply(
        lambda row: get_cluster_market_cap(
            cluster_lookup[row['cluster']],
            row['date']
        ),
        axis=1
    )

    return leaders

def construct_portfolio(
    year:int,
    freq:str,
    schema:str='equal',
    benchmark:str='^GSPC'
) -> Portfolio:
    """
    Build a portfolio object from cluster leaders.

    Args:
    year (int): The year to backtest.
    freq (str): Rebalancing frequency for the portfolio.
    schema (str): Weighting scheme for the portfolio, defaulting to equal weight.
    benchmark (str): Benchmark ticker symbol to compare against.

    Returns:
    Portfolio: A constructed portfolio containing the holding periods.
    """
    leaders = extract_leaders(year=year, freq=freq)
    period_end = create_end_date(year)
    rebalance_dates = sorted(leaders['rebalance_date'].dropna().unique())
    period_inputs = []
    for rebalance_date in rebalance_dates:
        period_df = leaders[
            (leaders['rebalance_date'] == rebalance_date)
            & (leaders['status'] == 'success')
        ].copy()

        if period_df.empty:
            continue

        # Use the scheduled date to group leaders, but use the latest valid
        # network date as the tradable start for the holding period.
        start = pd.Timestamp(period_df['date'].max())
        period_inputs.append((start, period_df))

    holding_periods = []
    for i, (start, period_df) in enumerate(period_inputs):
        end = (
            period_inputs[i + 1][0]
            if i < len(period_inputs) - 1
            else period_end
        )

        tickers = period_df['leader'].tolist()

        if not tickers:
            continue

        weights = weight_portfolio(
            tickers,
            schema=schema,
            market_caps=period_df.set_index('leader')['cluster_market_cap']
        )

        holding_periods.append(
            HoldingPeriod(
                period=(start, end),
                holdings=weights
            )
        )

    strategy_id = f'{schema}_{freq}'

    return Portfolio(holding_periods, strategy_id, benchmark)
