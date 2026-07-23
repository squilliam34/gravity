'''A portfolio class for analysis and tracking performance'''
from dataclasses import dataclass, field
import pandas as pd
from typing import Dict, Optional, Iterable
import yfinance as yf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from config import DATA_DIR

@dataclass
class HoldingPeriod:
    """
    Represents a single holding period within a portfolio.

    Attributes:
    period: A tuple of (start_timestamp, end_timestamp) for the holding period.
    holdings: Mapping of ticker symbols to portfolio weights for the period.
    benchmark_prices: Cached benchmark price series for the period (set during
      return calculation).
    benchmark_returns: Cached benchmark returns for the period (set during
      return calculation).
    portfolio_returns: Calculated portfolio returns time series for the period.
    """
    period: tuple[pd.Timestamp, pd.Timestamp]
    holdings: Dict[str, float]

    # Period only needs to hold returns. Global portfolio
    # will hold prices for all holdings across the whole
    # period. Then the period calculates its returns 
    benchmark_prices: Optional[pd.Series] = field(default=None, init=False, repr=False)
    benchmark_returns: Optional[pd.Series] = field(default=None, init=False, repr=False)

    portfolio_returns: Optional[pd.Series] = field(default=None, init=False, repr=False)

    def calculate_returns(
        self, 
        prices: pd.DataFrame, 
        benchmark_prices: pd.Series
    ) -> pd.Series:
        """
        Compute returns for the holdings in this period.

        This method extracts the price series for the tickers held during the
        period, computes daily percentage returns, aggregates them using the
        period weights to form a portfolio return series, and computes the
        benchmark returns for the same date range.

        Args:
        prices (pd.DataFrame): DataFrame of adjusted close prices indexed by date with
          columns for all tickers in the overall portfolio.
        benchmark_prices (pd.Series): Series of adjusted benchmark close prices indexed
          by date.

        Returns:
        pd.Series: The portfolio returns for this holding period.
        """
        start, end = self.period

        tickers = list(self.holdings.keys())
        prices = prices.loc[
            start:end,
            tickers
        ]

        self.returns = prices.pct_change().dropna()
        weights = pd.Series(self.holdings)
        self.portfolio_returns = (
            self.returns
            .mul(weights, axis=1)
            .sum(axis=1)
        )

        benchmark = benchmark_prices.loc[start:end]
        self.benchmark_returns = (
            benchmark
            .pct_change()
            .dropna()
        )

        return self.portfolio_returns

    def __str__(self) -> str:
        """
        Return a human-readable summary of the holding period.

        Shows the date range and each ticker with its weight formatted as a
        percentage.
        """

        start, end = self.period
        lines = [
            f'Holding Period: {start.date()} - {end.date()}',
            f'Holdings: {len(self.holdings)}',
            '-' * 40
        ]

        for ticker, weight in sorted(self.holdings.items()):
            lines.append(f'{ticker:<6} {weight:.2%}')

        return '\n'.join(lines)

class Portfolio:
    """
    Container for multiple `HoldingPeriod` objects representing a
    backtest/trading strategy over time.

    The `Portfolio` gathers prices for all tickers across the combined
    date range, computes concatenated portfolio and benchmark returns, and
    provides simple analysis helpers such as plotting and Sharpe ratio
    calculation.
    """

    def __init__(
        self,
        periods: Iterable['HoldingPeriod'],
        strategy_id : str,
        benchmark: str = '^GSPC'
    ) -> None:
        """
        Initialize the portfolio.

        Args:
        periods (Iterable[HoldingPeriod]): Iterable of `HoldingPeriod` instances defining the
          strategy across time.
        benchmark (str): Ticker symbol for the benchmark (default '^GSPC').
        """
        self.benchmark = benchmark
        self.periods = periods
        self.strategy_id = strategy_id

        self.start_date = min(
            p.period[0]
            for p in self.periods
        )

        self.end_date = max(
            p.period[1]
            for p in self.periods
        )

        # Store prices for every stock in the portfolio
        # across the period regardless of holding times
        self.prices = None
        self.benchmark_prices = None

        self.portfolio_returns = None
        self.benchmark_returns = None

    def load_prices(
        self, 
    ) -> pd.DataFrame:
        """
        Load (and cache) price series for all tickers and the benchmark.

        This method will look for cached CSV files under the configured
        `DATA_DIR` and, if missing, download adjusted close prices using
        `yfinance` and write them to cache.

        Args:
        benchmark (str): Benchmark ticker to download if benchmark cache is
          missing (default '^GSPC').

        Returns:
        pd.DataFrame: Adjusted close prices for all tickers across the
          portfolio's full date range.
        """

        if self.prices is not None:
            return self.prices

        price_dir = (
            DATA_DIR
            / 'portfolios'
            / f'{self.strategy_id}'
            / f'{pd.Timestamp(self.start_date).strftime('%Y-%m-%d')}_'
              f'{pd.Timestamp(self.end_date).strftime('%Y-%m-%d')}'
        )
        price_dir.mkdir(parents=True, exist_ok=True)
        price_path = price_dir / 'prices.csv'

        benchmark_dir = (
            DATA_DIR
            / 'benchmarks'
            / f'{self.benchmark}'
            / f'{pd.Timestamp(self.start_date).strftime('%Y-%m-%d')}_'
              f'{pd.Timestamp(self.end_date).strftime('%Y-%m-%d')}'
        )
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_path = benchmark_dir / 'benchmark.csv'

        # Check cache
        if price_path.exists(): 
            self.prices = pd.read_csv(
                price_path, 
                index_col=0, 
                parse_dates=True
            )

        else:
            tickers = set()
            for period in self.periods:
                tickers.update(period.holdings.keys())

            self.prices = yf.download(
                tickers,
                start=self.start_date,
                end=self.end_date,
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
                self.benchmark,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False
            )['Close']

            self.benchmark_prices.to_csv(benchmark_path)
        return self.prices

    def calculate_returns(self) -> pd.Series:
        """
        Compute and concatenate returns for all holding periods.

        The method calls each `HoldingPeriod.calculate_returns` to compute the
        per-period portfolio returns and then concatenates and sorts them to
        produce continuous portfolio and benchmark return series for the
        strategy.

        Returns:
        pd.Series: Concatenated portfolio returns indexed by date.
        """

        # concatenate period returns
        portfolio_returns = []
        benchmark_returns = []

        for period in self.periods:

            portfolio_returns.append(
                period.calculate_returns(
                    self.prices,
                    self.benchmark_prices
                )
            )

            benchmark_returns.append(
                period.benchmark_returns
            )

        self.portfolio_returns = (
            pd.concat(portfolio_returns)
            .sort_index()
        )

        self.benchmark_returns = (
            pd.concat(benchmark_returns)
            .sort_index()
        )

        return self.portfolio_returns

    def plot(self) -> None:
        """
        Plot cumulative growth of $1 for the portfolio and benchmark.

        Computes cumulative compounded equity curves from the return series
        and displays a matplotlib line chart comparing the portfolio to the
        benchmark.
        """

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
    ) -> float:
        """
        Compute the (annualized) Sharpe ratio for the portfolio.

        Args:
        risk_free_rate (float): Annual risk-free rate expressed as a decimal (e.g.
          0.02 for 2%).
        annualization (int): Number of trading periods per year (default 252).

        Returns:
        float: Annualized Sharpe ratio (mean excess return divided by
          standard deviation, scaled by sqrt(annualization)).
        """

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