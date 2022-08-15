from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

import numpy as np
import copy


class LSTM(BaseLayer):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(LSTM, self).__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._memorize = False
        self.hidden_state = np.zeros((1, hidden_size))
        self.cell_state = np.zeros((1, hidden_size))

        self.input_fcl = FullyConnected(self.input_size + self.hidden_size, 4 * self.hidden_size)
    
        self.forget_sigmoid = Sigmoid()
        self.input_sigmoid = Sigmoid()
        self.hidden_tanh = TanH()
        self.hidden_sigmoid = Sigmoid()

        self.hidden_updated_tanh = TanH()

        self.output_fcl = FullyConnected(self.hidden_size, self.output_size)
        self.output_sigmoid = Sigmoid()

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        output_tensor = np.empty((input_tensor.shape[0], self.output_size))
        if not self._memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))
            self.cell_state = np.zeros((1, self.hidden_size))
        self.input_tensor = input_tensor
    
        self.input_fcl_input_tensor = []

        self.forget_sigmoid_input_tensor = []
        self.input_sigmoid_input_tensor = []
        self.hidden_tanh_input_tensor = []
        self.hidden_sigmoid_input_tensor = []
    
        self.prev_cell_state = []
    
        self.hidden_updated_tanh_input_tensor = []
    
        self.output_fcl_input_tensor = []
        self.output_sigmoid_input_tensor = []

        for i, input_vector in enumerate(input_tensor):
            self.prev_cell_state.append(self.cell_state)
            self.input_fcl_input_tensor.append(np.c_[self.hidden_state, input_vector.reshape(1, -1)])

            input_fcl_output = self.input_fcl.forward(self.input_fcl_input_tensor[-1])
            self.forget_sigmoid_input_tensor.append(input_fcl_output[0, :self.hidden_size])
            self.input_sigmoid_input_tensor.append(input_fcl_output[0, self.hidden_size:2*self.hidden_size])
            self.hidden_tanh_input_tensor.append(input_fcl_output[0, 2*self.hidden_size:3*self.hidden_size])
            self.hidden_sigmoid_input_tensor.append(input_fcl_output[0, 3*self.hidden_size:])
    
            forget_gate = self.forget_sigmoid.forward(self.forget_sigmoid_input_tensor[-1])
            input_gate = self.input_sigmoid.forward(self.input_sigmoid_input_tensor[-1])
            hidden_state = self.hidden_tanh.forward(self.hidden_tanh_input_tensor[-1])
            self.cell_state = forget_gate * self.cell_state + input_gate * hidden_state
    
            self.hidden_updated_tanh_input_tensor.append(self.cell_state)

            self.hidden_state = self.hidden_updated_tanh.forward(self.cell_state) * self.hidden_sigmoid.forward(self.hidden_sigmoid_input_tensor[-1])

            self.output_fcl_input_tensor.append(self.hidden_state)
            self.output_sigmoid_input_tensor.append(self.output_fcl.forward(self.hidden_state))
            output_tensor[i] = self.output_sigmoid.forward(self.output_sigmoid_input_tensor[-1])
        return output_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        downstream_gradient = np.empty_like(self.input_tensor)
        hidden_gradient = np.zeros_like(self.hidden_state)
        cell_gradient = np.zeros_like(self.cell_state)
        self._gradient_weights = np.zeros_like(self.input_fcl.weights)
        output_gradient_weights = np.zeros_like(self.output_fcl.weights)

        for i, error_vector in reversed(list(enumerate(error_tensor))):
            self.output_sigmoid.forward(self.output_sigmoid_input_tensor[i])
            output_sigmoid_downstream_gradient = self.output_sigmoid.backward(error_vector)
    
            self.output_fcl.forward(self.output_fcl_input_tensor[i])
            output_fcl_downstream_gradient = self.output_fcl.backward(output_sigmoid_downstream_gradient)
            output_fcl_copy_downstream_gradient = hidden_gradient + output_fcl_downstream_gradient
            output_gradient_weights += self.output_fcl._gradient_weights
    
            hidden_sigmoid_error_tensor = output_fcl_copy_downstream_gradient * self.hidden_updated_tanh.forward(self.hidden_updated_tanh_input_tensor[i])
            hidden_updated_tanh_error_tensor = output_fcl_copy_downstream_gradient * self.hidden_sigmoid.forward(self.hidden_sigmoid_input_tensor[i])
            hidden_sigmoid_downstream_gradient = self.hidden_sigmoid.backward(hidden_sigmoid_error_tensor)
            hidden_updated_tanh_downstream_gradient = self.hidden_updated_tanh.backward(hidden_updated_tanh_error_tensor)
    
            summary_error_tensor = cell_gradient + hidden_updated_tanh_downstream_gradient

            forget_gate = self.forget_sigmoid.forward(self.forget_sigmoid_input_tensor[i])
            input_gate = self.input_sigmoid.forward(self.input_sigmoid_input_tensor[i])
            hidden_state = self.hidden_tanh.forward(self.hidden_tanh_input_tensor[i])

            input_and_hidden_error_tensor = summary_error_tensor
            input_error_tensor = input_and_hidden_error_tensor * hidden_state
            hidden_error_tensor = input_and_hidden_error_tensor * input_gate
            hidden_tanh_downstream_gradient = self.hidden_tanh.backward(hidden_error_tensor)
            input_sigmoid_downstream_gradient = self.input_sigmoid.backward(input_error_tensor)
    
            remaining_gate_error_tensor = summary_error_tensor
            cell_gradient = remaining_gate_error_tensor * forget_gate
            forget_sigmoid_error_tensor = remaining_gate_error_tensor * self.prev_cell_state[i]
            forget_sigmoid_downstream_gradient = self.forget_sigmoid.backward(forget_sigmoid_error_tensor)

            input_fcl_error_tensor = np.c_[forget_sigmoid_downstream_gradient,
                                           input_sigmoid_downstream_gradient,
                                           hidden_tanh_downstream_gradient, 
                                           hidden_sigmoid_downstream_gradient]

            self.input_fcl.forward(self.input_fcl_input_tensor[i])
            input_fcl_downstream_gradient = self.input_fcl.backward(input_fcl_error_tensor)
            downstream_gradient[i] = input_fcl_downstream_gradient[0, self.hidden_size:]
            hidden_gradient = input_fcl_downstream_gradient[0, :self.hidden_size]
            self._gradient_weights += self.input_fcl._gradient_weights

        if getattr(self, '_optimizer', False):
            self.input_fcl.weights = self._optimizer[0].calculate_update(self.input_fcl.weights, self._gradient_weights)
            self.output_fcl.weights = self._optimizer[1].calculate_update(self.output_fcl.weights, output_gradient_weights)
        return downstream_gradient

    def initialize(self, weights_initializer: object, bias_initializer: object):
        self.input_fcl.initialize(weights_initializer, bias_initializer)
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
        return self.input_fcl.weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self.input_fcl.weights = weights
