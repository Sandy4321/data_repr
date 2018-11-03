from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from collections import Counter

class Sampling(ABC):
    def split(self):
        pass

class SampleVarCat(Sampling):
    def __init__(self, data, n_samples):
        if n_samples >= len(data):
            raise ValueError('Sample number: {n} is greater than data points number: {d}'.format(n=n_samples, d=len(data)))
        c = Counter(data)
        self.cat_proportions = {k: v/len(data) for k, v in c.items()}
        self.target_proportions = {k: round(v * n_samples) for k, v in self.cat_proportions.items()}
        self.data = pd.DataFrame({'var': data})

    def sample(self):
        indices = []
        for k, v in self.target_proportions.items():
            s = self.data[self.data['var'] == k]
            indices.extend(s.sample(v).index.tolist())
        return indices