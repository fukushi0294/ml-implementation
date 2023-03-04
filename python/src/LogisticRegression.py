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
        self.W = optimize(self.y, self.x)

    def fit_transform(self, y: np.ndarray, x: np.ndarray):
        self.fit(y, x)
        self.transform(y, x)

    def predict(self, X: np.ndarray):
        p_mat = self.sigmoid(np.dot(X, self.W))
        return np.apply_along_axis(lambda x: x > self.thresh_hold, 0, p_mat)

    def log_likely_hood(self, X: np.ndarray, y: np.ndarray):
        XW = np.dot(X, self.W)
        mat_part = np.dot(y.transpose(), XW)
        log_part = np.log(1 + np.exp(XW))
        return np.sum(mat_part - log_part)

    def likely_hood(self, X: np.ndarray, y: np.ndarray):
        XW = np.dot(X, self.W)
        prob = self.sigmoid(XW)
        likely_p = np.power(prob, y)
        unlikely_p = np.power(1-prob, 1 - y)
        return np.prod(likely_p * unlikely_p)

    # TODO: move other class
    def gradient(self, f, x: np.ndarray):
        h = 1.0e-4
        grad = np.zeros_like(x)

        for idx in range(x.size):
            tmp = x[idx]
            x[idx] = tmp + h
            fxh1 = f(x)

            x[idx] = tmp - h
            fxh2 = f(x)
            grad = (fxh1 - fxh2) / (2*h)
            x[idx] = tmp
        return grad

    def numerical_grad(self, X: np.ndarray, y: np.ndarray):
        def loss_W(_): return - self.log_likely_hood(X, y)
        return self.gradient(loss_W, self.W)

    def simple_train(self, X: np.ndarray, y: np.ndarray):
        b_size = 2 if X.ndim == 1 else X.shape[1] + 1
        b = np.random.rand(b_size)
        lr = 0.1
        row_size = X.shape[0]
        one = np.ones((row_size, 1))
        x = np.append(one, X, axis=1)
        for _ in range(500):
            grad = self.numerical_grad(x, y)
            b -= lr * grad
            lr *= 0.9
        return b

    def sigmoid(self, x: np.ndarray):
        # suppress overflow
        signal = np.clip(x, -500, 500)
        e = np.exp(-signal)
        return 1 / (1 + e)

    def derivative_log_likely_hood(self, X: np.ndarray, y: np.ndarray):
        XW = np.dot(X, self.W)
        return np.dot(X.transpose(), (y - self.sigmoid(XW)))

    # minimize likely hood by SGD
    def sgd(self, X: np.ndarray, y: np.ndarray):
        self.W = np.random.randn(X.shape[1], 1)
        lr = 0.1
        grads = []
        likely_hoods = []
        # TODO: iteration count is static
        for _ in range(500):
            # Elastic Net
            grad = -self.derivative_log_likely_hood(X, y)
            grads.append(grad)
            self.W -= lr * grad
            likely_hoods.append(self.likely_hood(X, y))
            lr *= 0.9
        return self.W

    def l1_penalty_grad(self, b: np.ndarray):
        l1_norm = np.linalg.norm(b, ord=1)
        return np.array([l1_norm / _b for _b in b])

    def l2_penalty_grad(self, b: np.ndarray):
        l2_norm = np.linalg.norm(b, ord=2)
        return np.array([l2_norm / _b for _b in b])
