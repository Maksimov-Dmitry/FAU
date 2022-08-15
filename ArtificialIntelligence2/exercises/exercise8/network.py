'''
    TODO: Implement the functions: apply_network, forward, backward, update.
    You should use numpy but please do not use any libraries that are not part of the standard library.

    A network is defined by two parameters:
        weights: a list of numpy matrices, where weights[l][i,j] denotes the
                 weight of the edge from the j-th neuron in layer l to the
                 i-th neuron in layer l+1.
                 Note that the lecture defined w_{i,j} the other way around.
        biases:  a list of numpy vectors, where biases[l][i] denotes the
                 bias of the i-th neuron in layer l+1.

    Examples consist of tuples (input_, target) where:
        input_ is a numpy vector containing the input values
        target is a numpy vector containing the expected output values
'''

import numpy as np


def f(x):
    ''' the activation function - we use a fixed one for simplicity'''
    return np.tanh(x)

def df(x):
    ''' the derivative of the activation function '''
    return 1 - np.tanh(x)**2

def apply_network(weights, biases, input_):
    ''' Returns the output values of the network given the input input_ '''
    _, outputs = forward(weights, biases, input_)

    return outputs[-1]

def forward(weights, biases, input_):
    ''' Performs the forward propagation algorithm.
        Returns (in_, a)
        Where
            in_: contains the "activations before the activation function f was applied",
                 i.e., a[l+1][i] = f(in_[l][i]).
            a:   is the list of numpy vectors containing the neuron activations,
                 i.e., a[l][i] is the activation of the i-th neuron in layer l.
                 Note that a[0] should correspond to input_.
    '''
    in_ = []
    a = [input_]
    for layer_weights, layer_bias in zip(weights, biases):
        in_.append(np.dot(layer_weights, a[-1]) + layer_bias)
        a.append(f(in_[-1]))
    return in_, a

def backward(in_, a, weights, target):
    ''' Performs the back propagation algorithm.
        `in_` and `a` were computed by forward propagation (`forward`)
        and `target` describes the desired value.

        Returns deltas (Δ), where deltas[l][i] corresponds
        to the i-th neuron in layer l+1.

        Tip:
            if n is the number of layers - 1:
                deltas[n] = df(in_[n]) * (a[n+1] - target)
            else:
                deltas[n][j] = df(in_[n][j]) * Σ_i weights[n+1][i,j] * deltas[n+1][i]
    '''
    error_tensors = [df(in_[-1]) * (a[-1] - target)]
    for layer in reversed(range(len(weights) - 1)):
        error_tensors.append(df(in_[layer]) * np.dot(error_tensors[-1], weights[layer + 1]))
    error_tensors.reverse()
    return error_tensors

def update(weights, biases, delta, a, alpha):
    ''' Returns the updated weights and biases as a tuple (weights, biases).
        `delta` and `a` were computed by the previous functions.
        `alpha` is the step size - following the lecture, it is a negative value
        because we want to do gradient *descent*.

        Hints:
            - To use numpy operations, you may have to use `reshape` and `T` (transpose).
            - Don't forget that `weights` has the indices the other way around as in the lecture.
    '''
    updated_weights = []
    updated_biases = []
    for layer in range(len(delta)):
        updated_weights.append(weights[layer] + alpha * np.dot(delta[layer].reshape(-1, 1), a[layer].reshape((1, -1))))
        updated_biases.append(biases[layer] + alpha * delta[layer])

    return updated_weights, updated_biases

def train(shape, examples, alpha):
    ''' Initializes and trains a neural network.
        Arguments:
            shape:    a list of layer sizes. E.g. [2,3,1] would be a network with 2 input neurons,
                      a hidden layer consisting of 3 neurons and a single neuron in the output layer.
            examples: a list of training examples (pairs of input_ and target).
            alpha:    the step size (should be negative)
        Returns the weights and biases of the trained network
    '''
    # Randomly initialize weights and biases
    weights = []
    biases = []
    for i in range(len(shape)-1):
        weights.append(np.random.random((shape[i+1], shape[i])) - 0.5)
        biases.append(np.random.random(shape[i+1]) - 0.5)

    # train weights and biases with examples
    for input_, target in examples:
        in_, a = forward(weights, biases, input_)
        delta = backward(in_, a, weights, target)
        weights, biases = update(weights, biases, delta, a, alpha)

    return weights, biases
