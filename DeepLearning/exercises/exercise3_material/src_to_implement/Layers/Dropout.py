import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):

    def __init__(self, keep_prob: float):
        super(Dropout, self).__init__()
        self.keep_prob = keep_prob

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.dropout_mask = np.ones(input_tensor.shape)
        self.phase_keep_prob = 1
        if not self.testing_phase:
            self.phase_keep_prob = self.keep_prob
            self.dropout_mask = np.random.uniform(size=input_tensor.shape) < self.keep_prob
        return input_tensor * self.dropout_mask / self.phase_keep_prob

    def backward(self, error_tensor: float) -> np.ndarray:
        return error_tensor * self.dropout_mask / self.phase_keep_prob
