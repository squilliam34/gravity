'''A portfolio class for analysis and tracking performance'''

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