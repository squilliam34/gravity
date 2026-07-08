print('Importing libraries...')
from datetime import date
import pandas as pd
from pathlib import Path
import warnings
from tqdm import tqdm
import logging
import hdbscan

# Dimensionality reduction imports
import umap
from sklearn.manifold import SpectralEmbedding

# Personal modules
from data.data_loader import get_tickers
from src.distance.semantics.semantics_v2 import get_semantic_distances
from src.distance.factor_model.factor_model import load_factor_data, calculate_rolling_betas, compute_distances
from src.mass.mass import create_market_cap_df
from src.cluster import Cluster
print('Finished imports')

# Suppress FutureWarnings and YFinance warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

### Change dimensionality reduction model ###
MODEL = 'umap'

########################################
# PART 1: Import tickers
########################################

tickers = get_tickers('./data/csv/SP500.xlsx')
tickers.sort()

########################################
# PART 2: Semantic distances
########################################

semantic_path = Path('./data/S&P500/semantics_tenk/raw/semantic_distances.csv')

# --- SEMANTIC DISTANCES ---
if semantic_path.exists():
    print('Semantic distances already exist...')

    # Read in existing data to construct tickers that had valid 10-ks
    semantic_df = pd.read_csv('./data/S&P500/semantics_tenk/raw/semantic_distances.csv')
else:
    print('Computing semantic distances...')
    semantic_df = get_semantic_distances(tickers)
    semantic_df.to_csv(
        './data/S&P500/semantics_tenk/raw/semantic_distances.csv'
    )
    print('Semantic distances saved.')

########################################
# PART 3: Create 4-year intervals
########################################

intervals = []

for year in range(2000, date.today().year + 1, 4):

    start_date = f'{year}-01-01'

    if year + 1 < date.today().year:
        end_date = f'{year+1}-12-31'
    else:
        end_date = date.today().strftime('%Y-%m-%d')

    intervals.append(
        (start_date, end_date)
    )

########################################
# PART 4: Form Clusters
########################################

# I only want to have to calculate data sets for within a cluster
# Form the cluster then use the tickers in the cluster to compute data
cluster_path = Path('./data/clusters/clusters.csv')

if cluster_path.exists():
    print('CLusters already exist...')

    # Read in existing data to construct tickers that had valid 10-ks
    cluster_df = pd.read_csv('./data/S&P500/semantics_tenk/raw/semantic_distances.csv')
else:
    print('Clustering stocks...')

    # Read in cached embeddings to cluster with
    embeddings_df = pd.read_parquet('./data/cache/embeddings/gemini_item1_raw_cache.parquet')
    embeddings_df = embeddings_df.sort_values('ticker')

    tickers = embeddings_df['ticker'].to_list()
    X = np.vstack(
        embeddings_df['embedding'].values
    )
    n_components = 30
    state = 42

    if MODEL == 'umap':
        n_neighbors = 25
        min_dist = 0.05
        # UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            n_components=n_components, 
            min_dist=min_dist, 
            metric='cosine', 
            random_state=state
            )

    elif MODEL == 'spectral':
        reducer = SpectralEmbedding(
            n_components=n_components, 
            affinity='nearest_neighbors', 
            random_state=state)

    reduced_embeddings = reducer.fit_transform(X)

    # HDBScan
    min_cluster_size=5
    min_samples=3

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    clusterer.fit(reduced_embeddings)

    # Extract labels (-1 represents noise)
    labels = clusterer.labels_
    cluster_df = pd.DataFrame({'ticker': tickers, 'cluster': labels})

    cluster_df.to_csv(
        './data/S&P500/semantics_tenk/raw/semantic_distances.csv'
    )
    print('Clusters saved.')

# Assign Clusters
clusters = []
for i in set(cluster_df['cluster']):
    subset = cluster_df[cluster_df['cluster'] == i]
    tickers = subset['ticker'].to_list()
    clusters.append(Cluster(label=i, tickers=tickers))

########################################
# PART 5: Loop over intervals
########################################

for start_date, end_date in tqdm(
    intervals,
    desc='Processing intervals',
    unit='interval'
):

    factor_path = Path(f'./data/S&P500/factor_distances/factor_distance_{start_date}_{end_date}.csv')
    market_cap_path = Path(f'./data/S&P500/market_caps/market_caps_{start_date}_{end_date}.csv')

    tqdm.write(f'Processing {start_date} → {end_date}')

    ####################################
    # Factor distances
    ####################################

    if factor_path.exists():
        tqdm.write(f'Factor distances already exist: {start_date} → {end_date}')

    else:
        tqdm.write('Loading factor data...')
        factor_data = load_factor_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        betas = calculate_rolling_betas(
            data=factor_data,
            tickers=tickers
        )
        factor_distance = compute_distances(
            betas=betas
        )
        factor_distance.to_csv(
            f'./data/S&P500/factor_distances/factor_distance_{start_date}_{end_date}.csv'
        )
        tqdm.write('Factor distances saved.')

    ####################################
    # Market cap products
    ####################################

    if market_cap_path.exists():
        tqdm.write(f'Market caps already exist: {start_date} → {end_date}')

    else:
        tqdm.write('Calculating market caps...')
        market_caps = create_market_cap_df(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        market_caps.to_csv(
            f'./data/S&P500/market_caps/market_caps_{start_date}_{end_date}.csv'
        )
        tqdm.write('Market caps saved.')