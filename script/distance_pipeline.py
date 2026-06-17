print('Importing libraries...')
from datetime import date
import pandas as pd
from pathlib import Path
import warnings

# Personal modules
from data.data_loader import get_tickers
from src.distance.distance import get_lambda
print('Finished imports')

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

tickers = get_tickers('./data/csv/SP500.xlsx')

# Load semantic distances once
print('Loading semantic distances...')
semantic_distance = pd.read_csv(
    './data/S&P500/semantic_distances/semantic_distances.csv'
)

semantic_distance.set_index(
    ['stock_i', 'stock_j'],
    inplace=True
)

intervals = []

for year in range(2004, date.today().year + 1, 2):

    start_date = f'{year}-01-01'

    if year + 1 < date.today().year:
        end_date = f'{year+1}-12-31'
    else:
        end_date = date.today().strftime('%Y-%m-%d')

    intervals.append((start_date, end_date))

for start_date, end_date in intervals:

    distance_path = Path(f'./data/S&P500/distances/distance_{start_date}_{end_date}.csv')
    if distance_path.exists():
        print(f'Factor distances for {start_date} → {end_date} already exists')
        
    else:

        print('='*50)
        print(f'Processing {start_date} → {end_date}')
        print('='*50)
        
        # Read factor distances
        factor_distance = pd.read_csv(
            f'./data/S&P500/factor_distances/factor_distance_{start_date}_{end_date}.csv'
        )

        factor_distance['month'] = pd.PeriodIndex(
            factor_distance['month'],
            freq='M'
        )

        factor_distance.set_index(
            ['month', 'stock_i', 'stock_j'],
            inplace=True
        )

        # Create daily range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        daily_index = pd.date_range(
            start=start,
            end=end,
            freq='D'
        )

        # Get lambda values
        lam = get_lambda(
            start_date=start_date,
            end_date=end_date
        )

        # Create master dataframe
        pairs = pd.MultiIndex.from_product(
            [
                daily_index,
                tickers,
                tickers
            ],
            names=['date', 'stock_i', 'stock_j']
        )

        # Remove duplicates
        pairs = pairs[
            pairs.get_level_values('stock_i')
            <
            pairs.get_level_values('stock_j')
        ]

        df = pd.DataFrame(index=pairs).reset_index()

        # Join lambda
        df = df.join(
            lam,
            on='date'
        )

        # Join semantic distance
        df = df.join(
            semantic_distance,
            on=['stock_i', 'stock_j']
        )

        # Convert date → month
        df['month'] = df['date'].dt.to_period('M')

        # Join factor distances
        df = df.join(
            factor_distance,
            on=['month', 'stock_i', 'stock_j']
        )

        df.drop(
            columns='month',
            inplace=True
        )

        df.dropna(
            subset=['lambda', 'factor distance'],
            inplace=True
        )

        # Compute final distance
        df['Distance'] = (
            df['lambda']*df['factor distance']
            +
            (1-df['lambda'])*df['semantic distance']
        )

        df = df[
            ['date', 'stock_i', 'stock_j', 'Distance']
        ]

        df.set_index(
            ['date', 'stock_i', 'stock_j'],
            inplace=True
        )

        df.sort_index(inplace=True)

        # Save
        output_path = (
            f'./data/S&P500/distances/'
            f'distance_{start_date}_{end_date}.csv'
        )

        df.to_csv(output_path)

        print(f'Saved {output_path}')