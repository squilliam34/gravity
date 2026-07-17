import pandas as pd
import networkx as nx
from tqdm import tqdm
from src.cluster import Cluster
    
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
    Args:
    periods (list[tuple]): Rebalance periods as (start_date, end_date).
    clusters (list[Cluster]): List of Cluster objects.
    threshold (float): Optional edge weight threshold.
    force_recompute (bool): Recompute cached gravity data.

    Returns:
    pd.DataFrame: Cluster leaders by rebalance period.
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
                    'date': network_date,
                    'cluster': cluster.label,
                    'leader': leader,
                    'centrality': score,
                    'status': 'success'
                })

            except nx.NetworkXException:

                leaders.append({
                    'date': network_date,
                    'cluster': cluster.label,
                    'leader': None,
                    'centrality': None,
                    'status': 'centrality_failed'
                })

    return pd.DataFrame(leaders)