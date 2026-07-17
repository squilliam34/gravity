'''A portfolio class for analysis and tracking performance'''
from config import DATA_DIR
from pathlib import Path

class Portfolio:
    def __init__(
        self,
        name,
        initialization_date,
        holdings,
        benchmark='^GSPC'
    ):
        self.name = name
        self.initialization_date = initialization_date
        self.holdings = holdings
        self.benchmark = benchmark

        self.prices = None
        self.returns = None
        self.portfolio_returns = None
        self.benchmark_prices = None

    def load_prices(
        self,
        start_date,
        end_date
    ):

        tickers = list(self.holdings.values())

        self.prices = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True
        )['Close']

        return self.prices

    def calculate_returns(self):
        self.returns = (self.prices.pct_change().dropna())

        return self.returns

    def load_benchmark(self):

        if path.exists(): return pd.read_csv
        self.benchmark_prices = yf.download(
            self.benchmark,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True
        )['Close']