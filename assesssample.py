from abc import ABC, abstractmethod
from collections import Counter
from scipy.stats import chisquare, ks_2samp

class AssessSample(ABC):
    def __init__(self, sample, data):
        self.sample = sample
        self.data = data
        super().__init__()

    @abstractmethod
    def evaluate(self):
        pass

class AssessVarCat(AssessSample):
    """
    Evaluate the representativeness of a sample of a categorical variable versus its original data.
    """
    def __int__(self, sample, data):
        super().__init__(sample, data)
        self.sample = sample
        self.data = data


    def evaluate(self):
        """
        Perform a chisquare test
        :return: p value
        """
        data_size = len(self.data)
        data_counter = Counter(self.data)
        sample_counter = Counter(self.sample)
        categories = sorted(list(set(self.data)))
        observed = []
        expected_freq = []
        expected = []
        for cat in categories:
            observed.append(sample_counter[cat])
            expected_freq.append(data_counter[cat]/data_size)
            expected.append(round(data_counter[cat]/data_size * len(self.sample)))
        return chisquare(observed, expected)[1]

class AssessVarQuant(AssessSample):
    """
    Evaluate the representativeness of a sample variable versus its original data.
    """
    def __init__(self, sample, data):
        super().__init__(sample, data)
        self.sample = sample
        self.data = data

    def evaluate(self):
        """
        Perform a Kolmogorov-Smirnov test
        :return: p value
        """
        return ks_2samp(self.sample, self.data)[1]

