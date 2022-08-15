import numpy as np


class L1_Regularizer:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def calculate_gradient(self, weights: np.ndarray):
        return self.alpha * np.sign(weights)

    def norm(self, weights: np.ndarray) -> np.ndarray:
        return self.alpha * np.absolute(weights).sum()


class L2_Regularizer:

    def __init__(self, alpha: float):
        self.alpha = alpha

    def calculate_gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.alpha * weights

    def norm(self, weights: np.ndarray) -> np.ndarray:
        return self.alpha * np.square(weights).sum()
