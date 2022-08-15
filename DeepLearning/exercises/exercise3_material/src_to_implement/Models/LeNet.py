from NeuralNetwork import NeuralNetwork
from Optimization import *
from Layers import Conv, Flatten, FullyConnected, SoftMax, Initializers, ReLU


def build() -> NeuralNetwork:
    optimizer = Optimizers.Adam(5e-4, 0.9, 0.999)
    optimizer.add_regularizer(Constraints.L2_Regularizer(4e-4))
    net = NeuralNetwork(optimizer,
                        Initializers.He(),
                        Initializers.Constant(0.1))

    net.loss_layer = Loss.CrossEntropyLoss()

    # 1(3) x 28 x 28 -> 6 x 14 x 14
    net.append_layer(Conv.Conv((2, 2), (1, 5, 5), 6))
    net.append_layer(ReLU.ReLU())

    # 6 x 14 x 14 -> 16 x 7 x 7
    net.append_layer(Conv.Conv((2, 2), (6, 5, 5), 16))
    net.append_layer(ReLU.ReLU())

    # 16 x 7 x 7 -> 784
    net.append_layer(Flatten.Flatten())

    net.append_layer(FullyConnected.FullyConnected(784, 120))
    net.append_layer(ReLU.ReLU())

    net.append_layer(FullyConnected.FullyConnected(120, 84))
    net.append_layer(ReLU.ReLU())

    net.append_layer(FullyConnected.FullyConnected(84, 10))
    net.append_layer(SoftMax.SoftMax())

    return net
