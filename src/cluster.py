'''Cluster class to track stocks and perform analysis'''

class Cluster:
    
    def __init__(self, label:int, tickers:list[str], date:str, embeddings=None, market_caps=None, factor_data=None):

        self.label = label
        self.tickers = tickers
        self.date = date
        
        self.embeddings = embeddings
        self.market_caps = market_caps
        self.factor_data = factor_data

    def __str__(self):
        return f'Cluster {self.label}: {', '.join(self.tickers)}'

    def get_tickers(self):
        return self.tickers