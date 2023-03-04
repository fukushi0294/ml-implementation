import numpy as np
from abc import ABCMeta, abstractmethod

class Optimizer(metaclass= ABCMeta):
    def __init__(self, learning_rate: float) -> None:
        self.lr = learning_rate
        self.grads = np.array([[]])

    @abstractmethod
    def update(self, current: np.ndarray, grads: np.ndarray) -> np.ndarray:
        pass

class SGD(Optimizer):
    def update(self, current: np.ndarray, grads: np.ndarray) -> np.ndarray:
        np.append(self.grads, grads)
        updated = current - self.lr * grads
        # liner scheduler
        self.lr *= 0.9
        return updated

class Momentum(Optimizer):
    def __init__(self, learning_rate: float, momentum:float = 0.9) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v = None

    def update(self, current: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(current)
        
        np.append(self.grads, grads)
        self.v = self.momentum * self.v - self.lr * grads
        self.lr *= 0.9
        return current + self.v

class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float) -> None:
        super().__init__(learning_rate)
        self.h = 0

    def update(self, current: np.ndarray, grads: np.ndarray) -> np.ndarray:
        np.append(self.grads, grads)
        self.h += np.linalg.norm(grads, ord=2)
        return current - self.lr / self.h * grads
