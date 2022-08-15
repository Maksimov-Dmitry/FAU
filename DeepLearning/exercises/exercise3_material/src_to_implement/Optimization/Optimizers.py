import numpy as np
from Optimization.Constraints import L1_Regularizer, L2_Regularizer
from typing import Union


class Opimizer:

    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer: Union[L1_Regularizer, L2_Regularizer]):
        self.regularizer = regularizer


class Sgd(Opimizer):

    def __init__(self, learning_rate: float):
        super(Sgd, self).__init__()
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        regularizer_gradient = 0
        if getattr(self, 'regularizer', False):
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - self.learning_rate * gradient_tensor - self.learning_rate * regularizer_gradient


class SgdWithMomentum(Opimizer):

    def __init__(self, learning_rate: float, momentum_rate: float):
        super(SgdWithMomentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prev_gradient = 0

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        current_gradient = self.momentum_rate * self.prev_gradient - self.learning_rate * gradient_tensor
        self.prev_gradient = current_gradient
        regularizer_gradient = 0
        if getattr(self, 'regularizer', False):
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor + current_gradient - self.learning_rate * regularizer_gradient


class Adam(Opimizer):

    def __init__(self, learning_rate: float, mu: float, rho: float):
        super(Adam, self).__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.prev_v = 0
        self.prev_r = 0
        self.iteration = 0

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        self.iteration += 1
        current_v = self.mu * self.prev_v + (1 - self.mu) * gradient_tensor
        current_r = self.rho * self.prev_r + (1 - self.rho) * np.power(gradient_tensor, 2)
    
        self.prev_v = current_v
        self.prev_r = current_r
    
        current_v_corrected = current_v / (1 - self.mu ** self.iteration)
        current_r_corrected = current_r / (1 - self.rho ** self.iteration)
    
        regularizer_gradient = 0
        if getattr(self, 'regularizer', False):
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - self.learning_rate * current_v_corrected / np.clip(np.sqrt(current_r_corrected), np.finfo(float).eps, None) - self.learning_rate * regularizer_gradient
