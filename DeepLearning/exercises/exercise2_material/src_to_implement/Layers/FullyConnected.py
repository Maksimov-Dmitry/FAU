from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):

    def __init__(self, input_size: int, output_size: int):
        super(FullyConnected, self).__init__()
        self.weights = np.random.uniform(size=(input_size + 1, output_size))
        self.trainable = True
        self.fan_in = input_size
        self.fan_out = output_size

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor = np.c_[np.ones((input_tensor.shape[0], 1)), input_tensor]
        return np.dot(self.input_tensor, self.weights)
    
    def initialize(self, weights_initializer: object, bias_initializer: object) -> None:
        weights = weights_initializer.initialize((self.fan_in, self.fan_out), self.fan_in, self.fan_out)
        bias = bias_initializer.initialize((1, self.fan_out), self.fan_in, self.fan_out)
        self.weights = np.r_[bias, weights]
    
    @property
    def optimizer(self) -> object:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_object: object) -> None:
        self._optimizer = optimizer_object

    #dL/dX = dL/dY*WˆT
    #dL/dW = XˆT*dL/dY; XˆT - self.input_layer, dL/dY - upstream gradient = error_tensor
    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        backward_layer = np.dot(error_tensor, self.weights[1:].T)
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if getattr(self, '_optimizer', False):
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return backward_layer

    @property
    def gradient_weights(self) -> object:
        return self._gradient_weights
