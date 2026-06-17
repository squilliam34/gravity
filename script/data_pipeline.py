print('Importing libraries...')
from datetime import date
import pandas as pd
from pathlib import Path

# Personal modules
from data.data_loader import get_tickers
from src.distance.semantics.semantics import get_semantic_distances
from src.distance.factor_model.factor_model import load_factor_data, calculate_rolling_betas, compute_distances
from src.mass.mass import create_market_cap_df
import warnings
print('Finished imports')

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

########################################
# PART 1: Import tickers
########################################

tickers = get_tickers('./data/csv/SP500.xlsx')

########################################
# PART 2: Semantic distances
########################################

semantic_path = Path('./data/S&P500/semantic_distances/semantic_distances.csv')

# --- SEMANTIC DISTANCES ---
if semantic_path.exists():
    print('Semantic distances already exist...')
else:
    print('Computing semantic distances...')
    semantic_df = get_semantic_distances(tickers)
    semantic_df.to_csv(
        './data/S&P500/semantic_distances/semantic_distances.csv'
    )
    print('Semantic distances saved.')

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

for start_date, end_date in intervals:

    factor_path = Path(f'./data/S&P500/factor_distances/factor_distance_{start_date}_{end_date}.csv')
    market_cap_path = Path(f'./data/S&P500/market_caps/market_caps_{start_date}_{end_date}.csv')

    print()
    print('=' * 50)
    print(f'Processing {start_date} → {end_date}')
    print('=' * 50)

    ####################################
    # Factor distances
    ####################################

    if factor_path.exists():
        print(f'Factor distances for {start_date} → {end_date} already exists')
        
    else:
        print('Loading factor data...')
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
        print('Factor distances saved.')

    ####################################
    # Market cap products
    ####################################

    if market_cap_path.exists():
        print(f'Market caps for {start_date} → {end_date} already exists')
    else:
        print('Calculating market caps...')
        market_caps = create_market_cap_df(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        market_caps.to_csv(
            f'./data/S&P500/market_caps/market_caps_{start_date}_{end_date}.csv'
        )
        print('Market caps saved.')