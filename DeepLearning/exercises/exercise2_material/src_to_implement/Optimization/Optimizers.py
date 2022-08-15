import numpy as np


class Sgd:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:

    def __init__(self, learning_rate: float, momentum_rate: float):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prev_gradient = 0

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        current_gradient = self.momentum_rate * self.prev_gradient - self.learning_rate * gradient_tensor
        self.prev_gradient = current_gradient
        return weight_tensor + current_gradient


class Adam:

    def __init__(self, learning_rate: float, mu: float, rho: float):
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
        return weight_tensor - self.learning_rate * current_v_corrected / np.clip(np.sqrt(current_r_corrected), np.finfo(float).eps, None)
