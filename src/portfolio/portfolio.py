'''A portfolio class for analysis and tracking performance'''
from dataclasses import dataclass
import pandas as pd
from typing import Dict, Optional
import yfinance as yf

@dataclass(frozen=true)
class HoldingPeriod:
    period: tuple[pd.Timestamp, pd.Timestamp]
    holdings: Dict[str, float]

    prices: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    returns: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    benchmark_prices: Optional[pd.Series] = field(default=None, init=False, repr=False)
    benchmark_returns: Optional[pd.Series] = field(default=None, init=False, repr=False)

    portfolio_returns: Optional[pd.Series] = field(default=None, init=False, repr=False)
    def load_prices(self, benchmark='^GSPC'):
        start, end = self.period

        tickers = list(self.holdings.keys())

        self.prices = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )['Close']

        self.benchmark_prices = yf.download(
            benchmark,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )['Close']

        return self.prices

    def calculate_returns(self):
        if self.prices is None:
            raise ValueError("Prices have not been loaded.")

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
        self.states: states

    def calculate_returns(self):
        # concatenate state returns
        pass

    def plot(self):
        # plot full strategy
        pass

    def sharpe(self):
        # use the concate
        pass