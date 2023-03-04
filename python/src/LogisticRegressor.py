import numpy as np
from .common.math import sigmoid, gradient
from .common.optimizer import Optimizer

# This is basic implementation for logistic regression
# TODO: Assume input is one dimension. because use for DecisionTree
#       log(p/(1-p)) = b0 + b1 * x1 + b2 * x2^2 + ...
#      log likely hood
#          Π(p_i^y_i*(1-p_i)^(1-y_i)) = Y_transpose*X*B - Σ(1 + exp(X*B))
#      derivative log likely hood
#          (y_i - p_i) * X_i  (p = 1/(1+exp(-X*B)))


class LogisticRegressor:
    def __init__(self, optimizer: Optimizer) -> None:
        self.thresh_hold = 0.8
        self.optimizer = optimizer

    def predict(self, X: np.ndarray):
        p_mat = sigmoid(np.dot(X, self.W))
        return np.apply_along_axis(lambda x: x > self.thresh_hold, 0, p_mat)

    def log_likely_hood(self, X: np.ndarray, y: np.ndarray):
        XW = np.dot(X, self.W)
        mat_part = np.dot(y.transpose(), XW)
        log_part = np.log(1 + np.exp(XW))
        return np.sum(mat_part - log_part)

    def likely_hood(self, X: np.ndarray, y: np.ndarray):
        XW = np.dot(X, self.W)
        prob = sigmoid(XW)
        likely_p = np.power(prob, y)
        unlikely_p = np.power(1-prob, 1 - y)
        return np.prod(likely_p * unlikely_p)

    def numerical_grad(self, X: np.ndarray, y: np.ndarray):
        def loss_W(_): return - self.log_likely_hood(X, y)
        return gradient(loss_W, self.W)

    def derivative_log_likely_hood(self, X: np.ndarray, y: np.ndarray):
        XW = np.dot(X, self.W)
        return np.dot(X.transpose(), (y - sigmoid(XW)))

    def fit(self, X: np.ndarray, y: np.ndarray, iteration: int = 1000):
        self.W = np.random.randn(X.shape[1], 1)
        likely_hoods = []
        for _ in range(iteration):
            grad = -self.derivative_log_likely_hood(X, y)
            self.W = self.optimizer.update(self.W, grad)
            likely_hoods.append(self.likely_hood(X, y))
        return self.W

    def l1_penalty_grad(self, b: np.ndarray):
        l1_norm = np.linalg.norm(b, ord=1)
        return np.array([l1_norm / _b for _b in b])

    def l2_penalty_grad(self, b: np.ndarray):
        l2_norm = np.linalg.norm(b, ord=2)
        return np.array([l2_norm / _b for _b in b])
