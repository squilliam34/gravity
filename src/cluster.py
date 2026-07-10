'''Cluster class to track stocks and perform analysis'''
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from collections.abc import Callable
import numpy as np

# Personal modules
from src.distance.factor_model.factor_model import load_factor_data, calculate_rolling_betas, compute_distances
from src.mass.mass import create_market_cap_df
from src.distance.semantics.semantics_v2 import get_semantic_distances
from src.distance.distance import get_lambda

@dataclass
class State:
    """
    Store the cached data for a cluster over a specific date range.
    """
    # Date range - used to form the key
    start_date: str
    end_date: str

    # Path to csv directory in project that holds its data
    path: Path

    # Hold the data
    market_caps: object = None
    semantic_distances: object = None
    factor_distances: object = None
    weights: object = None
    distances: object = None
    gravity: object = None

    # Will turn false if there are not enough stocks with
    # data available for the state
    valid: bool = True
    reason: str | None = None

class Cluster:
    """
    Represent a named group of tickers with cached analysis state.
    """

    def __init__(self, label: int, tickers: list[str]) -> None:
        """Initialize a cluster with its label and ticker list.

        Args:
        label (int): The numeric identifier for the cluster.
        tickers (list[str]): The stock tickers belonging to the cluster.

        Returns:
        None
        """
        self.label = label
        self.tickers = tickers

        self.states = {}

    def __str__(self) -> str:
        """
        Return a human-readable description of the cluster.

        Args:
        None

        Returns:
        str: A string showing the cluster label and its tickers.
        """
        return f"Cluster {self.label}: {', '.join(self.tickers)}"

    def get_tickers(self) -> list[str]:
        """
        Return the list of tickers in the cluster.

        Args:
        None

        Returns:
        list[str]: The tickers in the cluster.
        """
        return self.tickers

    def get_state(self, start_date, end_date) -> State:
        """
        Retrieve or create the state object for a given date range.

        Args:
        start_date (str): The start date for the requested state.
        end_date (str): The end date for the requested state.

        Returns:
        State: The cached state object for the requested date range.
        """
        key = (start_date, end_date)

        if key not in self.states:
            # Make the desired directory if it doesn't exist yet
            path = Path(f'./data/clusters/{self.label}/{start_date}_{end_date}/')
            path.mkdir(parents=True, exist_ok=True)

            self.states[key] = State(
                start_date=start_date,
                end_date=end_date,
                path=path
            )

        return self.states[key]

    def _get_data(
        self,
        start_date: str,
        end_date: str,
        attribute: str,
        compute_fn: Callable[[], pd.DataFrame],
        force_recompute: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve data from cache or compute it on demand.

        Args:
        start_date (str): The start date of the requested data window.
        end_date (str): The end date of the requested data window.
        attribute (str): The state attribute to read or write.
        compute_fn (callable): A function that computes the data when it is not cached.
        force_recompute (bool): If True, indicates to ignore the existing data and recompute it.

        Returns:
        pd.DataFrame: The requested data as a DataFrame.
        """
        state = self.get_state(start_date=start_date, end_date=end_date)
        path = state.path / f'{attribute}.parquet'

        # Overwrite the existing data
        if force_recompute:
            data = compute_fn()

        # Check memory cache
        elif (data := getattr(state, attribute)) is not None:
            return data

        # Check disk cache
        elif path.exists():
            data = pd.read_parquet(path)

        # Compute from scratch
        else:
            data = compute_fn()

        # Handle failed computation
        if data.empty:
            state.valid = False
            state.reason = f'{attribute} unavailable or insufficient data'

        # Save and update state
        data.to_parquet(path)
        setattr(state, attribute, data)

        return data

        # Save and update state
        data.to_parquet(path)
        setattr(state, attribute, data)
        return data

    def get_factor_distances(
        self, 
        start_date, 
        end_date, 
        force_recompute = False
    ) -> pd.DataFrame:
        """
        Return factor-based distance data for the cluster.

        Args:
        start_date (str): The start date of the requested window.
        end_date (str): The end date of the requested window.
        force_recompute (bool): If True, indicates to ignore the existing data and recompute it.

        Returns:
        pd.DataFrame: Factor distance data for the cluster over the requested window.
        """
        return self._get_data(
            start_date,
            end_date,
            attribute='factor_distances',
            compute_fn=lambda: compute_distances(
                calculate_rolling_betas(
                    load_factor_data(
                        tickers=self.tickers,
                        start_date=start_date,
                        end_date=end_date,
                    ), self.tickers
                )
            ),
            force_recompute=force_recompute
        )

    def get_market_caps(
        self, 
        start_date, 
        end_date, 
        force_recompute = False
    ) -> pd.DataFrame:
        """
        Return the market capitalization data for the cluster.

        Args:
        start_date (str): The start date of the requested window.
        end_date (str): The end date of the requested window.
        force_recompute (bool): If True, indicates to ignore the existing data and recompute it.

        Returns:
        pd.DataFrame: Market cap data for the cluster over the requested window.
        """
        return self._get_data(
            start_date,
            end_date,
            attribute='market_caps',
            compute_fn=lambda: create_market_cap_df(
                tickers=self.tickers,
                start_date=start_date,
                end_date=end_date,
            ),
            force_recompute=force_recompute
        )

    def get_semantic_distances(
        self, 
        start_date, 
        end_date, 
        force_recompute = False
    ) -> pd.DataFrame:
        """
        Return semantic distance data for the cluster.

        Args:
        start_date (str): The start date of the requested window.
        end_date (str): The end date of the requested window.
        force_recompute (bool): If True, indicates to ignore the existing data and recompute it.

        Returns:
        pd.DataFrame: Semantic distance data for the cluster.
        """
        return self._get_data(
            start_date,
            end_date,
            attribute='semantic_distances',
            compute_fn=lambda: get_semantic_distances(
                self.tickers
            ),
            force_recompute=force_recompute
        )

    def get_weights(
        self,
        start_date, 
        end_date, 
        force_recompute = False
    ) -> pd.DataFrame:
        """
        Return semantic distance data for the cluster.

        Args:
        start_date (str): The start date of the requested window.
        end_date (str): The end date of the requested window.
        force_recompute (bool): If True, indicates to ignore the existing data and recompute it.

        Returns:
        pd.DataFrame: Vix data transformed to use as weights for distances for the cluster.
        """
        return self._get_data(
            start_date,
            end_date,
            attribute='weights',
            compute_fn=lambda: get_lambda(
                start_date=start_date,
                end_date=end_date
            ),
            force_recompute=force_recompute
        )

    def get_distances(
        self,
        start_date,
        end_date,
        force_recompute=False
    ):
        """
        Return distance data for the cluster.

        Args:
        start_date (str): The start date of the requested window.
        end_date (str): The end date of the requested window.
        force_recompute (bool): If True, indicates to ignore the existing data and recompute it.

        Returns:
        pd.DataFrame: Semantic distances and factor distances weighted by the weighting scheme.
        """
        return self._get_data(
            start_date,
            end_date,
            attribute='distances',
            compute_fn=lambda: self._calculate_distance(
                self._prepare_distance_inputs(
                    weights=self.get_weights(
                        start_date,
                        end_date,
                        force_recompute=force_recompute
                    ),
                    factor_distances=self.get_factor_distances(
                        start_date,
                        end_date,
                        force_recompute=force_recompute
                    ),
                    semantic_distances=self.get_semantic_distances(
                        start_date,
                        end_date,
                        force_recompute=force_recompute
                    ),
                )
            ),
            force_recompute=force_recompute
        )

    def get_gravity(
        self,
        start_date,
        end_date,
        force_recompute = False
    ) -> pd.DataFrame:
        """
        Return gravity data for the cluster.

        Args:
        start_date (str): The start date of the requested window.
        end_date (str): The end date of the requested window.
        force_recompute (bool): If True, indicates to ignore the existing data and recompute it.

        Returns:
        pd.DataFrame: The gravity data.
        """
        return self._get_data(
            start_date,
            end_date,
            attribute='gravity',
            compute_fn=lambda: self._calculate_gravity(
                self._prepare_gravity_inputs(
                    market_caps=self.get_market_caps(
                        start_date=start_date,
                        end_date=end_date,
                        force_recompute=force_recompute
                    ),
                    distances=self.get_distances(
                        start_date=start_date,
                        end_date=end_date,
                        force_recompute=force_recompute
                    )
                )
            ),
            force_recompute=force_recompute
        )

    def _prepare_distance_inputs(
        self,
        weights:pd.DataFrame, 
        factor_distances:pd.DataFrame, 
        semantic_distances:pd.DataFrame,
    ) -> pd.DataFrame:
        """ 
        Prepares the data prior to calculating distance.

        Args:
        weights (pd.DataFrame): The weights to weight each distance metric by.
        factor_distances (pd.DataFrame): The distance of factors between stocks across time.
        semantic_distances (pd.DataFrame): The distance in semantic meaning between stocks across time.

        Returns:
        pd.DataFrame: A DataFrame with all the data combined.
        """
        # Check if factor distance was created or not
        if factor_distances.empty: return pd.DataFrame()
        
        factor_distances = factor_distances.copy()
        semantic_distances = semantic_distances.copy()
        # Convert semantic distance matrix into a MultiIndex for easy alignment
 
        # Mask upper triangle (exclude diagonal + duplicates) 
        mask = np.triu(np.ones(semantic_distances.shape), k=1).astype(bool) 
        semantic_distances = ( 
            semantic_distances.where(mask) 
                .stack() 
                .to_frame('semantic distance') 
        ) 
        semantic_distances.index.names = ['stock_i', 'stock_j'] 

        daily_index = pd.DatetimeIndex(weights.index)

        # Create master dataframe
        pairs = pd.MultiIndex.from_product(
            [
                daily_index,
                self.tickers,
                self.tickers
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

        # Join lambda (the weights for )
        df = df.join(
            weights,
            on='date'
        )

        # Join semantic distances
        df = df.join(
            semantic_distances,
            on=['stock_i', 'stock_j']
        )

        # Convert date → month
        df['month'] = df['date'].dt.to_period('M')

        # Join factor distances
        df = df.join(
            factor_distances,
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
        return df


    def _calculate_distance(
        self, 
        df:pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate the final distance metric to be used in gravity calculations.

        Args:
        df (pd.DataFrame): A DataFrame with the combined factor distances and semantic distances.

        Returns:
        pd.DataFrame: A DataFrame containing the final distance value.
        """
        # Check for valid data
        if df.empty: return pd.DataFrame()

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
        return df

    def _prepare_gravity_inputs(self, market_caps, distances): 
        """
        Prepare all the data for gravity calculation.
        """ 
        # Check for valid data
        if distances.empty: return pd.DataFrame()

        market_caps = market_caps.copy()
        distances = distances.copy()

        # Ensure all data sources have the same index format
        market_caps = market_caps.reset_index()
        market_caps['date'] = pd.to_datetime(market_caps['date']).dt.tz_localize(None)
        market_caps = market_caps.set_index(['date','ticker']).sort_index()

        distances.reset_index(inplace=True)
        distances['date'] = pd.to_datetime(distances['date'])
        distances['date'] = (
            pd.to_datetime(distances['date'])
            .dt.tz_localize(None)
        )
        distances = distances.set_index(
            ['date', 'stock_i', 'stock_j']
        ).sort_index()

        # Ensure market_cap data has the same date range as distance data
        dates = distances.index.get_level_values('date').unique()
        market_caps = market_caps.loc[
            market_caps.index.get_level_values('date').isin(dates)
        ]

        # Convert market_caps df to multiindex that matches distance data
        idx = distances.index
        distances['mass_i'] = (
            market_caps.reindex(
                pd.MultiIndex.from_arrays(
                    [
                        idx.get_level_values('date'),
                        idx.get_level_values('stock_i')
                    ]
                )
            )['market_cap']
            .to_numpy()
        )

        distances['mass_j'] = (
            market_caps.reindex(
                pd.MultiIndex.from_arrays(
                    [
                        idx.get_level_values('date'),
                        idx.get_level_values('stock_j')
                    ]
                )
            )['market_cap']
            .to_numpy()
        )

        # Calculate mass products
        distances['mass_product'] = (
            distances['mass_i']
            * distances['mass_j']
        )
        return distances

    def _calculate_gravity(self, df:pd.DataFrame) -> pd.Series:
        """
        Calculates the gravity value.

        Args:
        df (pd.DataFrame): A MultiIndex DataFrame containing distance and mass data by date and stock.

        Returns:
        pd.DataFrame: A DataFrame of gravity values by date and stock.
        """
        # Check for valid data
        if df.empty: return pd.DataFrame()

        # Gravity calculation
        return (df['mass_product'] / df['Distance']).to_frame(name='Gravity')