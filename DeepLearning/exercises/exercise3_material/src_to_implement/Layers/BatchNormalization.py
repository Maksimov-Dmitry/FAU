import numpy as np
import copy

from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):

    def __init__(self, channels: int):
        super(BatchNormalization, self).__init__()
        self.trainable = True
        self.channels = channels
        self.initialize()
        self.batch_size = None
        self.image_size = None
        self.mean = None
        self.var = None
        self.alpha = 0.8

    def initialize(self, weight_initializer=None, bias_initializer=None) -> None:
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor = input_tensor
        input_tensor_2dim = input_tensor.copy()
        if input_tensor.ndim == 4:       
            self.batch_size = input_tensor.shape[0]
            self.image_size = input_tensor.shape[2:]
            input_tensor_2dim = self.reformat(input_tensor)

        self.mean_batch = input_tensor_2dim.mean(axis=0)
        self.var_batch = input_tensor_2dim.var(axis=0)
        if not self.testing_phase:
            mean_phase = self.mean_batch
            var_phase = self.var_batch
            if self.mean is None:
                self.mean = self.mean_batch
                self.var = self.var_batch
            else:
                self.mean = self.alpha * self.mean + (1 - self.alpha) * self.mean_batch
                self.var = self.alpha * self.var + (1 - self.alpha) * self.var_batch
        else:
            mean_phase = self.mean
            var_phase = self.var

        self.input_tensor_normalized = (input_tensor_2dim - mean_phase) / np.sqrt(np.clip(var_phase, np.finfo(float).eps, None))
        output_tensor_normalized = self.weights * self.input_tensor_normalized + self.bias

        if input_tensor.ndim == 4:
            output_tensor_normalized = self.reformat(output_tensor_normalized)
            self.input_tensor_normalized = self.reformat(self.input_tensor_normalized)
        return output_tensor_normalized

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        error_tensor_2dim = error_tensor.copy()
        input_tensor_normalized_2dim = self.input_tensor_normalized.copy()
        input_tensor_2dim = self.input_tensor.copy()
        if error_tensor.ndim == 4:
            error_tensor_2dim = self.reformat(error_tensor)
            input_tensor_normalized_2dim = self.reformat(self.input_tensor_normalized)
            input_tensor_2dim = self.reformat(self.input_tensor)

        downstream_gradient = compute_bn_gradients(error_tensor_2dim, input_tensor_2dim, self.weights, self.mean_batch, self.var_batch)
        if self.input_tensor.ndim == 4:
            downstream_gradient = self.reformat(downstream_gradient)
            
        self._gradient_weights = np.sum(error_tensor_2dim * input_tensor_normalized_2dim, axis=0)
        self._gradient_bias = error_tensor_2dim.sum(axis=0)
        if getattr(self, '_optimizer', False):
            self.weights = self._optimizer[0].calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer[1].calculate_update(self.bias, self._gradient_bias)
        return downstream_gradient

    def reformat(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.ndim == 4:
            return (tensor
                    .reshape(tensor.shape[0], tensor.shape[1], -1)
                    .transpose(0, 2, 1)
                    .reshape(-1, tensor.shape[1]))
        else:
            return (tensor
                    .reshape(self.batch_size, -1, tensor.shape[1])
                    .transpose(0, 2, 1)
                    .reshape(-1, tensor.shape[1], *self.image_size))

    @property
    def optimizer(self) -> object:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_object: object) -> None:
        self._optimizer = []
        self._optimizer.append(copy.deepcopy(optimizer_object))
        self._optimizer.append(copy.deepcopy(optimizer_object))

    @property
    def gradient_weights(self) -> np.ndarray:
        return self._gradient_weights

    @property
    def gradient_bias(self) -> np.ndarray:
        return self._gradient_bias
