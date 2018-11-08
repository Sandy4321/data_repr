import unittest
from variablesampling import SampleVarCat, SampleVarQuant
import numpy as np
import pandas as pd

class TestSampleVarCat(unittest.TestCase):
    def setUp(self):
        self.l = [2, 3, 1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 1, 1, 2]
        n_samples = 10
        self.svc = SampleVarCat(self.l, n_samples)
        self.svc_np = SampleVarCat(np.array(self.l), n_samples)
        self.svc_pd = SampleVarCat(pd.DataFrame({'x': self.l}).x, n_samples)

    def test_cat_proportions(self):
        self.assertEqual(self.svc.cat_proportions, {1: 0.4, 2: 0.4, 3: 0.2})
    def test_np_cat_proportions(self):
        self.assertEqual(self.svc_np.cat_proportions, {1: 0.4, 2: 0.4, 3: 0.2})
    def test_pd_cat_proportions(self):
        self.assertEqual(self.svc_pd.cat_proportions, {1: 0.4, 2: 0.4, 3: 0.2})
    def test_target_prop(self):
        self.assertEqual(self.svc.target_proportions, {1: 4.0, 2: 4.0, 3: 2.0})
    def test_np_target_prop(self):
        self.assertEqual(self.svc_np.target_proportions, {1: 4.0, 2: 4.0, 3: 2.0})
    def test_pd_target_prop(self):
        self.assertEqual(self.svc_pd.target_proportions, {1: 4.0, 2: 4.0, 3: 2.0})
    def test_value_error(self):
        with self.assertRaises(ValueError):
            SampleVarCat(self.l, 30)
    def test_sample(self):
        idx = self.svc.sample()
        self.assertListEqual(sorted([self.l[x] for x in idx]), sorted([2, 3, 1, 1, 2, 2, 3, 1, 1, 2]))
    def test_split(self):
        s, t = self.svc.split()
        self.assertListEqual(sorted(s), sorted([2, 3, 1, 1, 2, 2, 3, 1, 1, 2]))
        self.assertListEqual(sorted(t), sorted([2, 3, 1, 1, 2, 2, 3, 1, 1, 2]))

class TestSampleVarQUant(unittest.TestCase):
    def setUp(self):
        self.l = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
        self.svq = SampleVarQuant(self.l, 4)

    def test_sample(self):
        idx = self.svq.sample()
        self.assertListEqual(sorted([self.l[x] for x in idx]), sorted([0.1, 0.2, 0.3, 0.4]))

if __name__ == '__main__':
    unittest.main()
