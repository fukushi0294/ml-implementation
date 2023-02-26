import pandas as pd
import numpy as np
import math
from scipy import optimize

# This is basic implementation for logistic regression
# TODO: Assume input is one dimension. because use for DecisionTree
#       log(p/(1-p)) = b0 + b1 * x1 + b2 * x2^2 + ...
#      log likely hood
#          Π(p_i^y_i*(1-p_i)^(1-y_i)) = Y_transpose*X*B - Σ(1 + exp(X*B))
#      derivative log likely hood
#          (y_i - p_i) * X_i  (p = 1/(1+exp(-X*B)))


class LogisticRegression:
    def __init__(self) -> None:
        self.thresh_hold = 0.8

    def fit(self, y: np.ndarray, x: np.ndarray):
        self.y = y
        self.x = x

    def transform(self):
        self.b = optimize(self.y, self.x)

    def fit_transform(self, y: np.ndarray, x: np.ndarray):
        self.fit(y, x)
        self.transform(y, x)

    def predict(self, x: np.ndarray):
        # out: p/1-p
        out = np.apply_along_axis(
            lambda x: math.exp(np.dot(x, self.b)) + 1, 0, x)
        p_mat = np.fabs(np.reciprocal(out) - 1)
        return np.apply_along_axis(lambda x: x > self.thresh_hold, 0, p_mat)

    def log_likely_hood(self, y: np.ndarray, x: np.ndarray, b: np.ndarray):
        mat_part = np.dot(y.transpose(), np.dot(x, b))
        scala_part = 0
        for row in x:
            scala_part = + np.log(self.sigmoid(row, b))
        return mat_part + scala_part

    def likely_hood(self, y: np.ndarray, X: np.ndarray, b: np.ndarray):
        p = 1
        for idx, _y in enumerate(y):
            x = X[idx]
            prob = self.sigmoid(x, b)
            p = p * math.pow(prob, _y) * math.pow(1 - prob, 1 - _y)
        return p

    def sigmoid(self, x: np.ndarray, b: np.ndarray):
        # suppress overflow
        signal = np.clip(np.dot(x, b), -500, 500)
        e = np.exp(-signal)
        return 1 / (1 + e)

    def derivative_log_likely_hood(self, y: np.ndarray, X: np.ndarray, b: np.ndarray):
        sum = np.zeros(b.shape[0])
        for idx, _y in enumerate(y):
            x = X[idx]
            sum += (_y - self.sigmoid(x, b)) * x
        return sum

    # minimize likely hood by SGD
    def sgd(self, y: np.ndarray, X: np.ndarray):
        delta = 1.0e-14
        b_size = 2 if X.ndim == 1 else X.shape[1] + 1
        b = np.random.rand(b_size)
        lr = 0.1
        row_size = X.shape[0]
        one = np.ones((row_size, 1))
        x = np.hstack((one, X))
        grads = []
        likely_hoods = []
        # TODO: iteration count is static
        for _ in range(100):
            grad = -self.derivative_log_likely_hood(y, x, b)
            grads.append(grad)
            b -= lr * grad
            likely_hoods.append(self.likely_hood(y, x, b))
            lr *= 0.9
        return b
