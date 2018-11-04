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
        self.bins = bins
        self.n_samples = n_samples
        self.freq_bin, self.bin_edges = np.histogram(data, bins=bins, density=True)
        self.df = pd.DataFrame({'quantvar': data})


    def sample(self):
        a = list(self.data)
        edges = np.linspace(min(a) - abs(min(a)), max(a) + abs(min(a)), num=self.bins)
        dict_edges = {}
        dict_count = {}
        for i, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
            dict_edges[i] = (s, e)
            dict_count[i] = sum((s < a) & (a <= e)) / len(a)
        samples = []
        for i, freq in dict_count.items():
            # number of samples in this bin
            n = int(round(self.n_samples * freq))
            # boundaries of this bin
            s, e = dict_edges[i]
            vals = [a.index(x) for x in a if ((x > s) and (x <= e))]
            n_val = random.choices(vals, k=n)
            samples.append(n_val)
        samples = [x for sublist in samples for x in sublist]
        return samples

    def split(self):
        super().split()

