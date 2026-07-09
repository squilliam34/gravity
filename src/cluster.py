'''Cluster class to track stocks and perform analysis'''
from dataclasses import dataclass, field

@dataclass
class ClusterState:
    start_date: str
    end_date: str

    market_caps: object = None
    factor_data: object = None
    semantic_distances: object = None
    factor_distances: object = None
    gravity: object = None
    
class Cluster:
    
    def __init__(self, label:int, tickers:list[str], embeddings=None):

        self.label = label
        self.tickers = tickers
        
        self.embeddings = embeddings

        self.states = {}

    def __str__(self):
        return f'Cluster {self.label}: {', '.join(self.tickers)}'

    def get_tickers(self):
        return self.tickers

    def get_state(self, start_date, end_date):
        key = (start_date, end_date)

        if key not in self.states:
            self.states[key] = ClusterState(
                start_date=start_date,
                end_date=end_date
            )

        return self.states[key]