from Layers.Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor = input_tensor
        return np.maximum(input_tensor, 0)

    #dL/dX = dL/dY * g'(X): g'(X) = 1 if > 0 and 0 if < 0
    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        d_relu = error_tensor.copy()
        d_relu[self.input_tensor < 0] = 0
        return d_relu
