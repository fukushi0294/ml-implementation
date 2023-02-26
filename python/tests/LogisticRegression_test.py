import unittest
from python.src.LogisticRegression import LogisticRegression
import pandas as pd
import numpy as np


class TestLogisticRegression(unittest.TestCase):
    # def test_optimize(self):
    #     clf = LogisticRegression()
    #     df = pd.read_csv('./python/data/challenger_data.csv', index_col=False)
    #     df = df.dropna()
    #     y = df['Damage Incident'].to_numpy()
    #     X = df['Temperature'].to_numpy()
    #     X = X.reshape(X.shape[0], 1)
    #     b = clf.sgd(y, X)
    
    def test_sgd(self):
        clf = LogisticRegression()
        N = 100
        np.random.seed(0)
        X = np.random.randn(N, 2)
        # 5x + 3y = 1
        def f(x1, x2):
            return 5 * x1 + 3 * x2 - 1
        y = np.array([ 1 if f(x1, x2) > 0 else 0 for x1, x2 in X])
        b = clf.sgd(y, X)
        print(b)

if __name__ == '__main__':
    unittest.main()
