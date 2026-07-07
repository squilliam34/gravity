print('Importing libraries...')
from datetime import date
import pandas as pd
from pathlib import Path
import warnings
from tqdm import tqdm
import logging

# Personal modules
from data.data_loader import get_tickers
from src.distance.semantics.semantics_v2 import get_semantic_distances
from src.distance.factor_model.factor_model import load_factor_data, calculate_rolling_betas, compute_distances
from src.mass.mass import create_market_cap_df
print('Finished imports')

# Suppress FutureWarnings and YFinance warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

########################################
# PART 1: Import tickers
########################################

tickers = get_tickers('./data/csv/SP500.xlsx')
tickers.sort()

########################################
# PART 2: Semantic distances
########################################

semantic_path = Path('./data/S&P500/semantics_tenk/semantic_distances.csv')

# --- SEMANTIC DISTANCES ---
if semantic_path.exists():
    print('Semantic distances already exist...')

    # Read in existing data to construct tickers that had valid 10-ks
    semantic_df = pd.read_csv('./data/S&P500/semantics_tenk/semantic_distances.csv')
else:
    print('Computing semantic distances...')
    semantic_df = get_semantic_distances(tickers)
    semantic_df.to_csv(
        './data/S&P500/semantics_tenk/semantic_distances.csv'
    )
    print('Semantic distances saved.')

tickers = sorted(set(semantic_df['stock_i'])
    | set(semantic_df['stock_j']))

########################################
# PART 3: Create 2-year intervals
########################################

intervals = []

for year in range(2004, date.today().year + 1, 2):

    start_date = f'{year}-01-01'

    if year + 1 < date.today().year:
        end_date = f'{year+1}-12-31'
    else:
        end_date = date.today().strftime('%Y-%m-%d')

    intervals.append(
        (start_date, end_date)
    )

########################################
# PART 4: Loop over intervals
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