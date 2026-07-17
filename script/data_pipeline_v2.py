print('Importing libraries...')
from datetime import date
import pandas as pd
from pathlib import Path
import warnings
from tqdm import tqdm
import logging
import numpy as np
import time
import hdbscan
import umap.umap_ as umap

# Personal modules
from data.data_loader import get_tickers
from src.distance.semantics.semantics_v2 import get_semantic_distances
from src.distance.factor_model.factor_model import (
    load_factor_data, 
    calculate_rolling_betas, 
    compute_distances
)
from src.mass.mass import create_market_cap_df
from src.cluster import Cluster
print('Finished imports')

# Suppress FutureWarnings and YFinance warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

### Recompute existing gravity? ###
FORCE_RECOMPUTE = False

def build_clusters(year):

    cluster_path = Path(f'./data/clusters/{year}/clusters.csv')
    cluster_path.parent.mkdir(parents=True, exist_ok=True)

    if cluster_path.exists():
        return pd.read_csv(cluster_path)

    embeddings_df = pd.read_parquet(
        f'./data/cache/embeddings/{year}/gemini_item1_raw_cache.parquet'
    ).sort_values('ticker')

    tickers = embeddings_df['ticker'].tolist()

    X = np.vstack(embeddings_df['embedding'])

    X = umap.UMAP(
        n_neighbors=15,
        min_dist=0.0,
        n_components=10,
        metric='cosine',
        random_state=42,
    ).fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=1,
        prediction_data=True,
    )

    clusterer.fit(X)

    P = hdbscan.all_points_membership_vectors(clusterer)

    labels = P.argmax(axis=1)

    cluster_df = pd.DataFrame({
        'ticker': tickers,
        'cluster': labels
    })

    cluster_df.to_csv(cluster_path, index=False)

    return cluster_df

########################################
# PART 1: Create 5-year intervals
########################################

intervals = []

for year in range(2010, date.today().year + 1, 5):

    start_date = f'{year}-01-01'

    if year + 5 < date.today().year:
        end_date = f'{year+4}-12-31'
    else:
        end_date = date.today().strftime('%Y-%m-%d')

    intervals.append(
        (start_date, end_date)
    )

########################################
# PART 3: Loop over intervals
########################################

# Include a delay and retry loop incase the YFinance API gets overloaded
# Should work after first try if the API returns an error response
MAX_RETRIES = 5
RETRY_DELAY = 1

for start_date, end_date in tqdm(
    intervals,
    desc='Processing intervals',
    unit='interval'
):
    tqdm.write(f'Processing {start_date} → {end_date}')

    year = start_date[:4]
    
    # Call tickers by year
    tickers_csv = pd.read_csv(f'./data/csv/{year}/tickers.csv')
    tickers = tickers_csv['Ticker'].to_list()
    tickers.sort()

    # Don't need to check path since cache already handles this
    print(f'Computing semantic distances for {year}...')
    semantic_df = get_semantic_distances(tickers, year)
    
    cluster_df = build_clusters(year)
    clusters = []
    for label, group in cluster_df.groupby('cluster'):

        clusters.append(
            Cluster(
                label=label,
                tickers=group['ticker'].tolist()
            )
        )

    for cluster in tqdm(
        clusters,
        desc='Processing clusters',
        unit='cluster'
    ):

        for attempt in range(MAX_RETRIES):
            try:
                cluster.get_gravity(
                    start_date=start_date,
                    end_date=end_date,
                    force_recompute=FORCE_RECOMPUTE
                )
                break  # success

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    tqdm.write(
                        f'{cluster.label} failed '
                        f'({attempt+1}/{MAX_RETRIES}): {e}'
                    )
                    time.sleep(RETRY_DELAY)

                else:
                    tqdm.write(
                        f'{cluster.label} permanently failed: {e}'
                    )