import unittest
from python.src.LogisticRegression import LogisticRegression
import pandas as pd
import os


class TestLogisticRegression(unittest.TestCase):
    def test_optimize(self):
        clf = LogisticRegression()
        df = pd.read_csv('./python/data/challenger_data.csv', index_col=False)
        df = df.dropna()
        y = df['Damage Incident'].to_numpy()
        X = df['Temperature'].to_numpy()
        b = clf.optimize(y, X)

if __name__ == '__main__':
    unittest.main()