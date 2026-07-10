'''Cluster class to track stocks and perform analysis'''
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

# Personal modules
from src.distance.factor_model.factor_model import load_factor_data, calculate_rolling_betas, compute_distances
from src.mass.mass import create_market_cap_df
from src.distance.semantics.semantics_v2 import get_semantic_distances

@dataclass
class State:
    # Date range - used to form the key
    start_date: str
    end_date: str

    # Path to csv directory in project that holds its data
    path: Path

    # Hold the data
    market_caps: object = None
    factor_data: object = None
    semantic_distances: object = None
    factor_distances: object = None
    gravity: object = None
    
class Cluster:
    
    def __init__(self, label:int, tickers:list[str]):

        self.label = label
        self.tickers = tickers

        self.states = {}

    def __str__(self):
        return f'Cluster {self.label}: {', '.join(self.tickers)}'

    def get_tickers(self):
        return self.tickers

    def get_state(self, start_date, end_date):
        key = (start_date, end_date)

        if key not in self.states:
            self.states[key] = State(
                start_date=start_date,
                end_date=end_date,
                path=Path(f'./data/clusters/{self.label}/{start_date}_{end_date}/')
            )

        return self.states[key]

    def get_factor_data(self, start_date, end_date):
        state = self.get_state(start_date=start_date, end_date=end_date)

        # Check if the factor data already exists in the state
        factor_data = state.factor_data
        if factor_data: return factor_data

        else:
            # Check if it exists in the directory => can be loaded
            data_dir = state.path / 'factor_data.parquet'
            if data_dir.exists():
                factor_data = pd.read_parquet(data_dir)
                state.factor_data = factor_data
                return factor_data

            else:
                # Data doesn't exist, so it need to be calculated
                factor_data = load_factor_data(
                    tickers=self.tickers,
                    start_date=start_date,
                    end_date=end_date
                )
                betas = calculate_rolling_betas(
                    data=factor_data,
                    tickers=self.tickers
                )
                factor_data = compute_distances(
                    betas=betas
                )

                # Write to directory for future use
                factor_data.to_parquet(data_dir)
                state.factor_data = factor_data
                return factor_data

    def get_market_caps(self, start_date, end_date):
        state = self.get_state(start_date=start_date, end_date=end_date)

        # Check if the factor data already exists in the state
        market_caps = state.market_caps
        if market_caps: return market_caps

        else:
            # Check if it exists in the directory => can be loaded
            data_dir = state.path / 'market_caps.parquet'
            if data_dir.exists():
                market_caps = pd.read_parquet(data_dir)
                state.market_caps = market_caps
                return market_caps

            else:
                # Data doesn't exist, so it need to be calculated
                market_caps = create_market_cap_df(
                    tickers=self.tickers,
                    start_date=start_date,
                    end_date=end_date
                )

                # Write to directory for future use
                market_caps.to_parquet(data_dir)
                state.market_caps = market_caps
                return market_caps
    
    def get_semantic_distances(self, start_date, end_date):
        state = self.get_state(start_date=start_date, end_date=end_date)

        # Check if the factor data already exists in the state
        semantic_distances = state.semantic_distances
        if semantic_distances: return semantic_distances

        else:
            # Check if it exists in the directory => can be loaded
            data_dir = state.path / 'semantic_distances.parquet'
            if data_dir.exists():
                semantic_distances = pd.read_parquet(data_dir)
                state.semantic_distances = semantic_distances
                return semantic_distances

            else:
                # Data doesn't exist, so it need to be calculated
                semantic_distances = get_semantic_distances(self.tickers)

                # Write to directory for future use
                semantic_distances.to_parquet(data_dir)
                state.semantic_distances = semantic_distances
                return semantic_distances