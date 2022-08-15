from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

import numpy as np
import copy


class RNN(BaseLayer):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RNN, self).__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._memorize = False
        self.hidden_state = np.zeros((1, hidden_size))
        self.hidden_fcl = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.hidden_tanh = TanH()
        self.output_fcl = FullyConnected(self.hidden_size, self.output_size)
        self.output_sigmoid = Sigmoid()

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        output_tensor = np.empty((input_tensor.shape[0], self.output_size))
        if not self._memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))
        self.input_tensor = input_tensor
    
        self.hidden_fcl_input_tensor = []
    
        self.hidden_tanh_input_tensor = []
    
        self.output_fcl_input_tensor = []
    
        self.output_sigmoid_input_tensor = []

        for i, input_vector in enumerate(input_tensor):
            self.hidden_fcl_input_tensor.append(np.c_[self.hidden_state, input_vector.reshape(1, -1)])
            self.hidden_tanh_input_tensor.append(self.hidden_fcl.forward(self.hidden_fcl_input_tensor[-1]))
            self.hidden_state = self.hidden_tanh.forward(self.hidden_tanh_input_tensor[-1])
            self.output_fcl_input_tensor.append(self.hidden_state)
            self.output_sigmoid_input_tensor.append(self.output_fcl.forward(self.output_fcl_input_tensor[-1]))
            output_tensor[i] = self.output_sigmoid.forward(self.output_sigmoid_input_tensor[-1])
        return output_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        downstream_gradient = np.empty_like(self.input_tensor)
        hidden_gradient = np.zeros_like(self.hidden_state)
        self._gradient_weights = np.zeros_like(self.hidden_fcl.weights)
        output_gradient_weights = np.zeros_like(self.output_fcl.weights)
        for i, error_vector in reversed(list(enumerate(error_tensor))):
            self.output_sigmoid.forward(self.output_sigmoid_input_tensor[i])
            output_sigmoid_downstream_gradient = self.output_sigmoid.backward(error_vector)
    
            self.output_fcl.forward(self.output_fcl_input_tensor[i])
            output_fcl_downstream_gradient = self.output_fcl.backward(output_sigmoid_downstream_gradient)
            output_fcl_copy_downstream_gradient = hidden_gradient + output_fcl_downstream_gradient
            output_gradient_weights += self.output_fcl._gradient_weights
    
            self.hidden_tanh.forward(self.hidden_tanh_input_tensor[i])
            hidden_tanh_downstream_gradient = self.hidden_tanh.backward(output_fcl_copy_downstream_gradient)

            self.hidden_fcl.forward(self.hidden_fcl_input_tensor[i])
            hidden_fcl_downstream_gradient = self.hidden_fcl.backward(hidden_tanh_downstream_gradient)

            downstream_gradient[i] = hidden_fcl_downstream_gradient[0, self.hidden_size:]
            hidden_gradient = hidden_fcl_downstream_gradient[0, :self.hidden_size]
            self._gradient_weights += self.hidden_fcl._gradient_weights

        if getattr(self, '_optimizer', False):
            self.hidden_fcl.weights = self._optimizer[0].calculate_update(self.hidden_fcl.weights, self._gradient_weights)
            self.output_fcl.weights = self._optimizer[1].calculate_update(self.output_fcl.weights, output_gradient_weights)
        return downstream_gradient

    def initialize(self, weights_initializer: object, bias_initializer: object):
        self.hidden_fcl.initialize(weights_initializer, bias_initializer)
        self.output_fcl.initialize(weights_initializer, bias_initializer)

    @property
    def memorize(self) -> object:
        return self._memorize

    @memorize.setter
    def memorize(self, bool: bool) -> None:
        self._memorize = bool

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
    def weights(self) -> np.ndarray:
        return self.hidden_fcl.weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self.hidden_fcl.weights = weights
