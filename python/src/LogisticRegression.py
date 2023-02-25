import pandas as pd
import numpy as np
import math
from scipy import optimize

# This is basic implementation for logistic regression
# TODO: Assume input is one dimension. because use for DecisionTree
#       log(p/(1-p)) = b0 + b1 * x1 + b2 * x2^2 + ...
#      Likely hood
#          Π(p_i^y_i*(1-p_i)^(1-y_i)) = Y_transpose*X*B - Σ(1 + exp(X*B))
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
        out = np.apply_along_axis(lambda x: math.exp(np.dot(x, self.b)) + 1, 0, x)
        p_mat = np.fabs(np.reciprocal(out) - 1)
        return np.apply_along_axis(lambda x: x > self.thresh_hold, 0, p_mat)
    
    def likely_hood(self, y: np.ndarray, x: np.ndarray):
        mat_part = np.multiply(y.transpose(), np.multiply(x, self.b))
        scala_part = 0
        for row in x:
            scala_part =- (1 + math.exp(np.linalg.norm(row * self.b)))
        return np.linalg.norm(mat_part) + scala_part

    # return function obejct to optimize later
    def derivative_likely_hood(self, y: np.ndarray, x: np.ndarray, b: np.ndarray):
        def derivative_likely_hood_function(input_b, idx_in_b: int):
            _b = np.copy(b)
            _b[idx_in_b] = input_b
            row_size = x.shape[0]
            one = np.ones((row_size,1))
            _x = np.hstack((one, x.reshape(row_size, 1)))
            constants =  np.dot(y.transpose(), np.dot(_x, _b))
            scala_part = 0
            for row in _x:
                tmp = math.exp(np.dot(row, _b))
                scala_part =+ (input_b * tmp)/ (1 + tmp)
            return constants - scala_part
        return derivative_likely_hood_function

    # 1. 対数尤度関数の微分の連立方程式を立てる
    # 2. 目的変数のうち一つb_jを選びb_j以外を定数として偏微分する
    # 3. Newton法を用いて連立方程式を解く
    # 4. 3の結果を2の定数として利用し, 残りのbについて2を再び行う
    # 5. 2~4をbが更新されなくなるまで繰り返す
    def optimize(self, y: np.ndarray, x: np.ndarray):
        delta = 1.0e-8
        b_size = 2 if x.ndim == 1 else x.shape[1] + 1
        b = np.ones(b_size)
        b_new = np.ones(b_size)
        while True:
            for j, _ in enumerate(b):
                derivative_likely_hood_f = self.derivative_likely_hood(y, x, b)
                b_new[j] = optimize.newton(derivative_likely_hood_f, b[j], args = (j,))
            if np.linalg.norm(b - b_new) < delta:
                break
            else:
                b = b_new
        return b_new
