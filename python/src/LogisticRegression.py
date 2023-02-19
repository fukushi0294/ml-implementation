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
    def __init__(self):
        self.b = np.zeros(3)

    def fit(self, y: np.ndarray, x: np.ndarray):
        self.y = y
        self.x = x

    # 1. 対数尤度関数の微分の連立方程式を立てる
    # 2. 目的変数のうち一つb_jを選びb_j以外を定数として偏微分する
    # 3. Newton法を用いて連立方程式を解く
    # 4. 3の結果を2の定数として利用し, 残りのbについて2を再び行う
    # 5. 2~4をbが更新されなくなるまで繰り返す
    def likely_hood(self, y: np.ndarray, x: np.ndarray):
        # get matrix part
        mat_part = np.multiply(y.transpose(), np.multiply(x, self.b))
        scala_part = 0
        for row in x:
            scala_part =- (1 + math.exp(np.linalg.norm(row * self.b)))
        return np.linalg.norm(mat_part) + scala_part

    # return function obejct to optimize later
    def derivative_likely_hood(self, y: np.ndarray, x: np.ndarray, idx: int):
        def derivative_likely_hood_function(b):
            constants =  np.linalg.norm(y.transpose() * x * b)
            scala_part = 0
            for row in x:
                tmp = math.exp(np.linalg.norm(row * self.b))
                scala_part =+ (row[idx] * tmp)/ (1 + tmp)
            return constants - scala_part
        return derivative_likely_hood_function

    def optimize(self, y: np.ndarray, x: np.ndarray):
        for i, b in enumerate(self.b):
            derivative_likely_hood_f = self.derivative_likely_hood(y, x, i)
            root = optimize.newton(derivative_likely_hood_f, 1.5)
