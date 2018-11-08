import pandas as pd
import numpy as np
from assesssample import AssessVarCat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

FILENAMES = ['abalone', 'annealing', 'auto_mpg', 'credit_screening', 'thyroid', 'automobile']

class Model():
    def __init__(self, FILENAMES, train=False, folder='data/intermediate/'):
        self.filenames = FILENAMES
        self.folder = folder
        if train:
            self.df = self.parse_df(self.filenames)
            self.df.to_csv(folder+'data_features.csv', index=False)

        else:
            self.df = pd.read_csv(folder+'data_features.csv')

        # self.train, self.test = self.split(self.df)

    def parse_df(self, filenames):
        """
        Load data and targets and store computed metrics into a dataframe
        :param filenames: list of string
        :return: pandas DataFrame
        """
        for i, filename in enumerate(filenames):
            data = pd.read_csv(self.folder + filename + '.csv', header=None)
            target = pd.read_csv(self.folder + filename + '.target', header=None)
            if i == 0:
                df = self.data_features(data.values, target.values.T)
            else:
                df = pd.concat([df, self.data_features(data.values, target.values.T)])
            print("Parsing %s: %d rows" % (filename, len(df)))
        return df

    def data_features(self, a, target):
        """
        Compute the number of unique values, and its mean, std, max, min
         from each columns of the array a
        :param a: numpy array
        :param target: numpy array
        :return: pandas DataFrame
        """
        mean_vals = []
        std_vals = []
        min_vals = []
        max_vals = []
        nb_unique_vals = []
        for i in range(a.shape[1]):
            # count the number of unique values
            vals = np.unique(a[:, i])
            nb_unique_vals.append(len(vals))
            # frequency of each values
            vals_dict = {x: np.sum(a[:, i] == x) for x in vals}
            mean_vals.append(np.mean(list(vals_dict.values())))
            std_vals.append(np.std(list(vals_dict.values())))
            min_vals.append(np.min(list(vals_dict.values())))
            max_vals.append(np.max(list(vals_dict.values())))
        return pd.DataFrame({'count_cat': nb_unique_vals,
                             'mean_count_cat': mean_vals,
                             'std_count_cat': std_vals,
                             'min_count_cat': min_vals,
                             'max_count_cat': max_vals,
                             'target': target[0].tolist()})

    def split(self, df):
        """
        Split the dataframe into a training and testing sets and checks target proportion
        :param df:
        :return: train, test dataframes
        """
        n_samples = round(len(df)*0.2)
        df.index = df.index.reindex(list(range(len(df))))[0]

        idx = df.sample(n_samples).index
        test_cond = df.index.isin(idx)
        test = df[test_cond].copy()
        train = df[~test_cond].copy()

        if (test.target.mean() == train.target.mean()):
            return train, test

    def train(self):
        """
        Train a logistic regression model and save it.
        :return: test accuracy
        """
        train, test = self.split(self.df)
        if train:
            lr = LogisticRegression()
            lr.fit(train.iloc[:, train.columns != 'target'].values, train.target.values)
            acc = accuracy_score(test.target, lr.predict(test.iloc[:, test.columns != 'target'].values))
            print("Test accuracy = %.2f" % acc)
            joblib.dump(lr, 'models/probvar.joblib')