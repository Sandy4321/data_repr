import unittest
from assesssample import AssessVarCat, AssessVarQuant, AssessCombVar
import numpy as np
import pandas as pd

class TestAssessVarCat(unittest.TestCase):
    def setUp(self):
        sample = np.array([7, 1, 7, 5, 7, 5, 5, 5, 2, 6, 0, 4, 4, 2, 0, 7, 7, 0, 5, 1])
        data = np.array([5, 7, 1, 3, 2, 7, 2, 0, 6, 6, 0, 7, 5, 7, 2, 7, 5, 7, 1, 8, 0, 6,
       1, 0, 1, 5, 9, 8, 7, 2, 7, 9, 7, 2, 7, 2, 4, 6, 3, 4, 5, 2, 0, 8,
       9, 2, 5, 1, 3, 0, 7, 4, 5, 1, 7, 5, 3, 3, 7, 1, 5, 4, 7, 7, 5, 2,
       6, 1, 4, 3, 5, 2, 9, 7, 6, 2, 4, 0, 3, 8, 5, 0, 0, 7, 7, 7, 3, 7,
       6, 7, 4, 9, 3, 4, 2, 8, 3, 4, 8, 0])
        self.varstat = AssessVarCat(sample, data)
        self.varstat_nodiff = AssessVarCat(sample,sample)

    def test_evaluate_true(self):
        self.assertEqual(self.varstat.evaluate(), 0.41452452181510796)

    def test_evaluate_false(self):
        self.assertNotEqual(self.varstat_nodiff, 0.41452452181510796)

    def test_nodiff_evaluate_true(self):
        self.assertEqual(self.varstat_nodiff.evaluate(), 1.)

class TestAssessVarQuant(unittest.TestCase):
    def setUp(self):
        sample = np.array([ 1.98458368,  1.22897571,  1.4328828 ,  1.69074792,  2.10167531,
        0.04840399,  2.1144082 ,  1.94282632, -0.16313886,  1.65358132,
        1.41155834,  0.5511495 ,  0.47065943,  0.92669783, -0.15328514,
        2.3028377 ,  2.8104927 ,  1.29511344,  2.8501273 , -0.76528535,
        0.07090746, -0.20779187,  0.9382775 ,  1.66128298,  0.50466159,
        0.19092708,  1.83773906,  2.82319106,  0.86253472,  0.16711107,
        0.04288426,  0.46137096,  2.09181067,  2.26423137,  2.41642692,
        0.48875985,  0.36742194,  1.15611909,  1.91768425,  0.26907645,
        0.19520347,  0.00482773, -0.56712005,  2.12950342,  3.08143309,
        2.09882771,  1.02471695,  2.99756255,  1.37932405,  0.16186917])
        data = np.array([-2.82991984,  1.17417035,  1.6515517 ,  0.68169849, -0.43633256,
        0.69131153, -2.69899192,  0.73328896,  1.94040966, -1.39414739,
       -0.84638785,  0.23578379, -2.02696313, -0.05007842, -0.82301484,
        0.60564362,  0.35778404, -1.00301429,  0.33132248,  0.91489521,
       -1.21773794, -1.35003744, -0.3803252 , -0.47723858,  0.80458893,
       -0.56354362, -0.44245326,  0.2086552 , -1.56767267, -1.11319539,
       -0.75251908, -0.30532658,  0.13962957, -1.38879737, -0.36869415,
       -0.23893132,  1.82212507,  2.24875678, -0.45256624,  3.08263286,
       -2.04158893,  0.28389643,  0.34995399, -0.12794829, -1.73176907,
        0.55402506, -0.0278963 , -1.00573244,  0.69403693,  0.92338499,
        0.04247006, -1.16725074,  0.09898859,  1.55460551,  0.931606  ,
        0.1990677 ,  0.01859002,  0.7017637 , -1.22769434,  0.07413306,
        0.28568497,  0.06184572, -0.79108461, -0.5237906 , -1.6983684 ,
        0.23098035,  0.25833321, -0.0598486 , -1.28374213, -0.66661254,
        0.71451891, -0.3507532 , -0.21051453, -2.86706224,  1.50838967,
        0.08070375, -0.57405817,  0.93165299,  2.23667849, -0.36690574,
        0.36395714, -0.45697441,  0.65852886,  1.17367648,  0.15031648,
       -0.00918671, -0.14815375,  0.7484252 ,  2.13224601,  0.49105916,
       -1.32411308, -0.12558985,  1.6862414 ,  0.58988467, -1.04881587,
        1.1928156 , -0.63027608, -0.10521538,  0.90466037,  1.22779325])
        self.varstat = AssessVarQuant(sample, data)
        self.varstat_nodiff = AssessVarQuant(sample, sample)

    def test_evaluate_true(self):
        self.assertEqual(self.varstat.evaluate(), 8.805408314769023e-06)

    def test_evaluate_false(self):
        self.assertNotEqual(self.varstat_nodiff.evaluate(), 8.805408314769023e-06)

    def test_nodiff_evaluate_true(self):
        self.assertEqual(self.varstat_nodiff.evaluate(), 1.)

class TestAssessCombVar(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv('../data/iris.csv')
        sample = data.sample(frac=0.3, random_state=123)
        self.varstat = AssessCombVar(sample, data, target_col='iris_class')
        self.unweighted_varstat = AssessCombVar(sample, data)

    def test_get_weights(self):
        self.assertEqual(self.varstat.weights, [1,1,1,1,10])
        self.assertNotEqual(self.unweighted_varstat.weights, [1,1,1,1,10])

    def test_evaluate(self):
        self.assertEqual(self.varstat.evaluate(), 0.5712377451855659)
        self.assertNotEqual(self.unweighted_varstat.evaluate(), 0.5712377451855659)

    def test_compute(self):
        self.assertEqual(self.varstat.pvals, [0.8613322468322858,0.9997295732811864,0.924128808645564,0.9481696969052422,
                                       0.2817692890949582])

if __name__ == '__main__':
    unittest.main()