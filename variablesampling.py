from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from collections import Counter
import random

class Sampling(ABC):
    def split(self):
        pass

class SampleVarCat(Sampling):
    def __init__(self, data, n_samples):
        self.data = data
        if n_samples >= len(data):
            raise ValueError('Sample number: {n} is greater than data points number: {d}'.format(n=n_samples, d=len(data)))
        c = Counter(data)
        self.cat_proportions = {k: v/len(data) for k, v in c.items()}
        self.target_proportions = {k: round(v * n_samples) for k, v in self.cat_proportions.items()}
        self.df = pd.DataFrame({'var': data})

    def sample(self):
        indices = []
        for k, v in self.target_proportions.items():
            s = self.df[self.df['var'] == k]
            indices.extend(s.sample(v).index.tolist())
        return indices

    def split(self):
        idx = self.sample()
        non_idx = list(range(len(self.data)))
        [non_idx.remove(i) for i in idx]
        s = [self.data[i] for i in idx]
        t = [self.data[i] for i in non_idx]
        return s, t

class SampleVarQuant(SampleVarCat):
    def __init__(self, data, n_samples, bins=10):
        self.data = data
        self.n_samples = n_samples
        self.freq_bin, self.bin_edges = np.histogram(data, bins=bins, density=True)
        self.df = pd.DataFrame({'quantvar': data})

    def sample(self):
        """get most probable indices samples according to the column values distribution"""
        counts_bin = np.round(self.freq_bin / np.sum(self.freq_bin) * self.n_samples)
        bin_edges_windows = np.array([self.bin_edges[0:len(self.bin_edges) - 1], self.bin_edges[1:len(self.bin_edges)]])
        steps = np.repeat(np.arange(len(counts_bin))[counts_bin > 0.], counts_bin[counts_bin > 0.].astype(int))
        print(steps)
        indexes = []
        for c,i in enumerate(steps):
            idx = self.df.quantvar.index.where((self.df.quantvar > bin_edges_windows[0, i]) &
                                      (self.df.quantvar <= bin_edges_windows[1, i])).tolist()
            idx = [int(x) for x in idx if not np.isnan(x)]
            print(idx)
            print(counts_bin[c])
            if idx and counts_bin[c]:
                indexes.append(np.random.choice(idx, size=int(counts_bin[c]), replace=False).astype(int).tolist())

        print(indexes)
        return indexes

    def split(self):
        super().split()

