import unittest
import pandas as pd
from probingvariables import Model


class TestModel(unittest.TestCase):
    def setUp(self):
        self.expected_df = pd.DataFrame({
            'count_cat': [4, 2, 2],
            'mean_count_cat': [1, 2, 2],
            'std_count_cat': [0, 0, 1],
            'min_count_cat': [1, 2, 1],
            'max_count_cat': [1, 2, 3],
            'target': [0,1,1]
        })

        self.splitting_df = pd.DataFrame({
            'vals':   [1,1,1,1,1,1,1,1,1,1],
            'target': [0,0,0,0,0,1,1,1,1,1]
        })

    def test_parse(self):
        model_train = Model(['test'], train=True, folder='../data/test/')
        df = model_train.parse_df(['test'])
        self.assertTrue((df.values == self.expected_df.values).all())

    def test_load(self):
        model_load = Model(['test'], folder='../data/test/')
        self.assertTrue((model_load.df.values == self.expected_df.values).all())

    def test_split(self):
        model_load = Model(['test'], folder='../data/test/')
        train, test = model_load.split(self.splitting_df)
        self.assertEqual(train.vals.sum(), 8)
        self.assertEqual(train.target.mean(), 0.5)
        self.assertEqual(test.vals.sum(), 2)
        self.assertEqual(test.target.mean(), 0.5)

    def test_train(self):
        df = pd.read_csv('../data/test/test_features.csv')

        model_load = Model(['test'], folder='../data/test/')
        train, test = model_load.split(df)
        acc = model_load.train(train, test)
        self.assertEqual(acc, 1.0)

if __name__ == '__main__':
    unittest.main()