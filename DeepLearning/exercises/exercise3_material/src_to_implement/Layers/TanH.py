from Layers.Base import BaseLayer
import numpy as np


class TanH(BaseLayer):

    def __init__(self):
        super(TanH, self).__init__()

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.forward_output = np.tanh(input_tensor)
        return self.forward_output.copy()

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        return error_tensor * (1 - np.power(self.forward_output, 2))
