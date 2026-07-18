'''A portfolio class for analysis and tracking performance'''
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=true)
class HoldingPeriod:
    period: tuple[pd.Timestamp, pd.Timestamp]
    holdings: Dict[str, float]

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