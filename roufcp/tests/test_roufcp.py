###########################################
# TESTING THE PYTHON PACKAGE RoufCP
###########################################

import numpy as np
import unittest
from roufcp import roufCP

class TestroufCP(unittest.TestCase):
    def test_univariate(self):
        X = np.random.randn(50)
        cp = roufCP(delta = 3, w = 3)
        result = cp.fit(X, moving_window=5, method = 'ttest', k = 2)
        self.assertTrue('cp' in result)
        self.assertTrue('exponential_entropy' in result)
        self.assertTrue(isinstance(result['cp'], list))
        self.assertTrue(isinstance(result['exponential_entropy'], (np.ndarray, np.generic) ))

    def test_multivariate(self):
        X = np.array(np.random.randn(100, 4))
        X[40:, :] += 3
        X[75:, :] -= 5
        mu = 1
        sigma = np.random.randn(100, 100)
        sigma = np.transpose(sigma) @ sigma
        cp = roufCP(delta = 5, w = 5)
        y = cp.fit(X, moving_window=10, k = 5)
        result = cp.hypothesis_test(y['cp'], y['exponential_entropy'][y['cp']], mu, sigma, a_delta = (10 ** 0.5) )
        self.assertTrue('Individual p value' in result)
        self.assertTrue('Joint p value' in result)
