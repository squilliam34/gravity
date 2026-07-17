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