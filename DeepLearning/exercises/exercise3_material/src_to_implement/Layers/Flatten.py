from Layers.Base import BaseLayer
import numpy as np


class Flatten(BaseLayer):

    def __init__(self):
        super(Flatten, self).__init__()
        self.shape = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        return error_tensor.reshape(self.shape)
