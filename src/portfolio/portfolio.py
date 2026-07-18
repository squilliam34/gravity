'''A portfolio class for analysis and tracking performance'''
from dataclasses import dataclass, field
import pandas as pd
from typing import Dict, Optional
import yfinance as yf
from pathlib import Path
from config import DATA_DIR
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

PERIODS = [
    ('2010-01-01', '2014-12-31'),
    ('2015-01-01', '2019-12-31'),
    ('2020-01-01', '2024-12-31'),
    ('2025-01-01', date.today().strftime('%Y-%m-%d')),
]


def get_valid_portfolio_period(start_date):
    """
    Return the predefined portfolio period containing start_date.
    """

    start_date = pd.Timestamp(start_date)

    for period_start, period_end in PERIODS:

        period_start = pd.Timestamp(period_start)
        period_end = pd.Timestamp(period_end)

        if period_start <= start_date <= period_end:
            return (
                period_start.strftime('%Y-%m-%d'),
                period_end.strftime('%Y-%m-%d')
            )

    raise ValueError(
        f'No valid portfolio period found for {start_date.date()}.'
    )

@dataclass
class HoldingPeriod:
    period: tuple[pd.Timestamp, pd.Timestamp]
    holdings: Dict[str, float]

    prices: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    returns: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    benchmark_prices: Optional[pd.Series] = field(default=None, init=False, repr=False)
    benchmark_returns: Optional[pd.Series] = field(default=None, init=False, repr=False)

    portfolio_returns: Optional[pd.Series] = field(default=None, init=False, repr=False)

    def load_prices(self, benchmark='^GSPC'):
        period_start, period_end = get_valid_portfolio_period(self.period[0])

        period_str = f'{pd.Timestamp(period_start).date()}_{pd.Timestamp(period_end).date()}'

        cache_dir = (
            DATA_DIR
            / 'portfolio'
            / period_str
            / 'prices'
        )

        cache_dir.mkdir(parents=True, exist_ok=True)

        price_path = cache_dir / 'prices.csv'
        benchmark_path = cache_dir / 'benchmark.csv'

        # Check cache
        if price_path.exists(): 
            self.prices = pd.read_csv(
                price_path, 
                index_col=0, 
                parse_dates=True
            )

        else:
            tickers = list(self.holdings.keys())

            self.prices = yf.download(
                tickers,
                start=period_start,
                end=period_end,
                auto_adjust=True,
                progress=False
            )['Close']

            self.prices.to_csv(price_path)

        if benchmark_path.exists():

            self.benchmark_prices = pd.read_csv(
                benchmark_path,
                index_col=0,
                parse_dates=True
            ).squeeze('columns')

        else:
            self.benchmark_prices = yf.download(
                benchmark,
                start=period_start,
                end=period_end,
                auto_adjust=True,
                progress=False
            )['Close']

            self.benchmark_prices.to_csv(benchmark_path)
        return self.prices

    def calculate_returns(self):
        if self.prices is None:
            raise ValueError('Prices have not been loaded.')

        self.returns = self.prices.pct_change().dropna()

        weights = pd.Series(self.holdings)

        self.portfolio_returns = (
            self.returns
            .mul(weights, axis=1)
            .sum(axis=1)
        )

        self.benchmark_returns = (
            self.benchmark_prices
            .pct_change()
            .dropna()
        )

        return self.portfolio_returns

class Portfolio:
    def __init__(
        self,
        states,
        benchmark='^GSPC'
    ):
        self.benchmark = benchmark
        self.states = states

        self.portfolio_returns = None
        self.benchmark_returns = None

    def calculate_returns(self):
        # concatenate state returns
        portfolio_returns = []
        benchmark_returns = []

        for state in self.states:

            # load prices if needed
            if state.prices is None:
                state.load_prices(self.benchmark)

            # calculate returns if needed
            if state.portfolio_returns is None:
                state.calculate_returns()

            portfolio_returns.append(state.portfolio_returns)
            benchmark_returns.append(state.benchmark_returns)

        self.portfolio_returns = (
            pd.concat(portfolio_returns)
            .sort_index()
        )

        self.benchmark_returns = (
            pd.concat(benchmark_returns)
            .sort_index()
        )

        return self.portfolio_returns

    def plot(self):
        # plot full strategy
        if self.portfolio_returns is None:
            self.calculate_returns()

        portfolio_equity = (
            1 + self.portfolio_returns
        ).cumprod()

        benchmark_equity = (
            1 + self.benchmark_returns
        ).cumprod()

        plt.figure(figsize=(10, 5))

        plt.plot(
            portfolio_equity,
            label='Portfolio'
        )

        plt.plot(
            benchmark_equity,
            label=self.benchmark
        )

        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Growth of $1')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.show()

    def sharpe(
        self,
        risk_free_rate: float = 0.0,
        annualization: int = 252
    ):
        # use the concate
        if self.portfolio_returns is None:
            self.calculate_returns()

        excess_returns = (
            self.portfolio_returns
            - risk_free_rate / annualization
        )

        return (
            np.sqrt(annualization)
            * excess_returns.mean()
            / excess_returns.std()
        )