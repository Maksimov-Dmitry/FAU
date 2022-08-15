import numpy as np
import copy
import pickle


def save(filename: str, net: object) -> None:
    with open(filename, 'wb') as file:
        pickle.dump(net, file)


def load(filename: str, data_layer: object) -> object:
    with open(filename, 'rb') as file:
        net = pickle.load(file)
        net.data_layer = data_layer
    return net


class NeuralNetwork:

    def __init__(self, optimizer: object, weights_initializer: object, bias_initializer: object):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.layers = []
        self.loss = []
        self.data_layer = None
        self.loss_layer = None

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['data_layer']
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.data_layer = None

    def forward(self) -> np.ndarray:
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        regularization_loss = 0
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if self.optimizer.regularizer and layer.trainable:
                regularization_loss += self.optimizer.regularizer.norm(layer.weights)
        return self.loss_layer.forward(input_tensor, label_tensor) + regularization_loss

    def backward(self) -> None:
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer: object) -> None:
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations: int) -> None:
        self.phase = 'train'
        for i in range(iterations):
            print('#' * 100)
            print(f'Iteration: {i+1} of {iterations}')
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor: np.ndarray) -> np.ndarray:
        self.phase = 'test'
        for layer in self.layers:
            print(layer)
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    @property   
    def phase(self) -> str:
        return self.phase

    @phase.setter
    def phase(self, phase: str) -> None:
        for layer in self.layers:
            if phase == 'train':
                layer.testing_phase = False
            else:
                layer.testing_phase = True
