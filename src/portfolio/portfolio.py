'''A portfolio class for analysis and tracking performance'''
from config import DATA_DIR
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import date


@dataclass(frozen=true)
class PortfolioState:
    dates: date
    holdings: Dict[str, float]

class Portfolio:
    def __init__(
        self,
        states,
        benchmark='^GSPC'
    ):
        self.benchmark = benchmark
        self.states: states

        self.returns = None
        self.portfolio_returns = None
        self.benchmark_prices = None

    def calculate_returns(
        self,
        prices
    ):

        portfolio_returns = []

        for i, state in enumerate(self.states):

            start = state.date
            if i < len(self.states)-1:
                end = self.states[i+1].date
            else:
                end = prices.index[-1]

            period_prices = prices.loc[start:end]
            weights = pd.Series(state.holdings)
            returns = (
                period_prices
                .pct_change()
                .mul(weights)
                .sum(axis=1)
            )

            portfolio_returns.append(returns)

        self.portfolio_returns = pd.concat(portfolio_returns)

        return self.portfolio_returns