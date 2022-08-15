import numpy as np
import copy


class NeuralNetwork:

    def __init__(self, optimizer: object, weights_initializer: object, bias_initializer: object):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.layers = []
        self.loss = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self) -> np.ndarray:
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return self.loss_layer.forward(input_tensor, label_tensor)

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
        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
