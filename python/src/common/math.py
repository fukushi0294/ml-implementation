import numpy as np


def gradient(f, x: np.ndarray):
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


def sigmoid(x: np.ndarray):
    # suppress overflow
    signal = np.clip(x, -500, 500)
    e = np.exp(-signal)
    return 1 / (1 + e)
