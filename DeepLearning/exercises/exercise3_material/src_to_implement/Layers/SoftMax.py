from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    
    def __init__(self):
        super(SoftMax, self).__init__()

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        exp_tensor = np.exp(input_tensor - np.max(input_tensor))
        self.softmax_tensor = exp_tensor/exp_tensor.sum(axis=1, keepdims=True)
        return self.softmax_tensor.copy()

    #ds/dx = s(x) * (1-s(x)) if i=j; -s(x)*s(x) if i!=j
    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        tensor1 = np.einsum('ij,ik->ijk', self.softmax_tensor, self.softmax_tensor)
        tensor2 = np.einsum('ij,jk->ijk', self.softmax_tensor, np.eye(self.softmax_tensor.shape[1], self.softmax_tensor.shape[1]))
        dSoftmax = tensor2 - tensor1
        dz = np.einsum('ijk,ik->ij', dSoftmax, error_tensor)
        return dz
