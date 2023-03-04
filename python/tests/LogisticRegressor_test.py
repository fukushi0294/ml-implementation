import unittest
from python.src.LogisticRegressor import LogisticRegressor
import pandas as pd
import numpy as np
import python.src.common.optimizer as opt


class TestLogisticRegression(unittest.TestCase):
    def load_test_data(self, N=100):
        np.random.seed(0)
        X = np.random.randn(N, 2)

        def f(x1, x2):
            return 5 * x1 + 3 * x2 - 1
        y = np.array([1 if f(x1, x2) > 0 else 0 for x1, x2 in X]
                     ).reshape(-1, 1)
        one = np.ones((N, 1))
        X = np.hstack((one, X))
        return X, y

    def test_sgd(self):
        clf = LogisticRegressor(optimizer=opt.SGD(learning_rate=0.1))
        X, y = self.load_test_data(100)
        trained_W = clf.fit(X, y)
        print(trained_W)

    def test_adagrad(self):
        clf = LogisticRegressor(optimizer=opt.AdaGrad(learning_rate=0.1))
        X, y = self.load_test_data(100)
        trained_W = clf.fit(X, y)
        print(trained_W)

    def test_momentum(self):
        clf = LogisticRegressor(optimizer=opt.Momentum(learning_rate=0.01))
        X, y = self.load_test_data(100)
        trained_W = clf.fit(X, y)
        print(trained_W)

    def test_likely_hood(self):
        clf = LogisticRegressor(optimizer=opt.SGD(learning_rate=0.1))
        X, y = self.load_test_data(100)
        clf.W = np.array([[-1, 3, 5]]).transpose()
        lh = clf.likely_hood(X, y)
        log_lh = clf.log_likely_hood(X, y)
        dlog_lh = clf.derivative_log_likely_hood(X, y)
        self.assertTrue(np.isscalar(lh))
        self.assertTrue(np.isscalar(log_lh))
        self.assertFalse(np.isscalar(dlog_lh))


if __name__ == '__main__':
    unittest.main()
