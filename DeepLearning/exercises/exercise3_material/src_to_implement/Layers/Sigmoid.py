from Layers.Base import BaseLayer
import numpy as np


class Sigmoid(BaseLayer):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.forward_output = 1 / (1 + np.exp(-input_tensor))
        return self.forward_output.copy()

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        return error_tensor * self.forward_output * (1 - self.forward_output)
