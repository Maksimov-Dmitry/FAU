from Layers.Base import BaseLayer
from typing import Union
import numpy as np
from scipy.signal import correlate, convolve
from math import ceil
import copy


class Conv(BaseLayer):

    def __init__(self, stride_shape: Union[tuple[int], tuple[int, int]], convolution_shape: Union[tuple[int, int], tuple[int, int, int]], num_kernels: int):
        super(Conv, self).__init__()
        self.trainable = True
        self.weights = np.random.uniform(size=(num_kernels, *convolution_shape))
        self.bias = np.random.uniform(size=num_kernels)
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.fan_in = np.prod(np.array(convolution_shape))
        self.fan_out = np.prod(np.array((num_kernels, *convolution_shape[1:])))

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor = input_tensor
        convolution_matrix_full_shape = (1 + (np.array(input_tensor.shape[2:]) - 1)//np.array(self.stride_shape))
        convolution_matrix_full = np.zeros((input_tensor.shape[0], self.num_kernels, *convolution_matrix_full_shape))
        for image_index, image in enumerate(input_tensor):
            for kernel_index, kernel in enumerate(self.weights):
                convolution_matrix_full[image_index, kernel_index] = self.bias[kernel_index]
                for channel_index, channel in enumerate(image):
                    convolution_matrix_kernel_channel = correlate(channel, kernel[channel_index], mode='same', method='direct')
                    if len(self.stride_shape) == 1:
                        convolution_matrix_kernel_channel = convolution_matrix_kernel_channel[::self.stride_shape[0]]
                    else:
                        convolution_matrix_kernel_channel = convolution_matrix_kernel_channel[::self.stride_shape[0], ::self.stride_shape[1]]
                    convolution_matrix_full[image_index, kernel_index] += convolution_matrix_kernel_channel

        return convolution_matrix_full

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        convolution_matrix_full = np.zeros(self.input_tensor.shape)
        gradient_weights = np.zeros_like(self.weights)
        for image_index, image in enumerate(error_tensor):
            for kernel_index, kernel in enumerate(self.weights):
                for kernel_channel_index, kernel_channel in enumerate(kernel):
                    strided_downstream_gradient = np.zeros((np.array(image[kernel_index].shape) - 1) * np.array(self.stride_shape) + 1)
                    if len(self.stride_shape) == 1:
                        strided_downstream_gradient[::self.stride_shape[0]] = image[kernel_index].copy()
                        P = [0]
                    else:
                        strided_downstream_gradient[::self.stride_shape[0], ::self.stride_shape[1]] = image[kernel_index].copy()
                        P = [0, 0]

                    for i in range(len(P)):
                        P[i] = self.input_tensor[image_index, kernel_channel_index].shape[i] - strided_downstream_gradient.shape[i] 
                        P[i] = (P[i] - ceil(P[i] / 2), ceil(P[i] / 2))
                    strided_padded_downstream_gradient_layer = np.pad(strided_downstream_gradient, P, mode='constant')
                    convolution_matrix_full[image_index, kernel_channel_index] += convolve(strided_padded_downstream_gradient_layer, kernel_channel, mode='same', method='direct')
                    for i in range(len(P)):
                        P[i] = kernel_channel.shape[i] + self.input_tensor[image_index, kernel_channel_index].shape[i] - 1 - strided_downstream_gradient.shape[i]
                        P[i] = (P[i] - ceil(P[i] / 2), ceil(P[i] / 2))
                    strided_padded_downstream_gradient_weights = np.pad(strided_downstream_gradient, P, mode='constant')
                    gradient_weights[kernel_index, kernel_channel_index] += correlate(self.input_tensor[image_index, kernel_channel_index], strided_padded_downstream_gradient_weights, mode='valid', method='direct')
        self._gradient_weights = gradient_weights
        self._gradient_bias = np.sum(error_tensor, axis=tuple(i for i in range(len(error_tensor.shape)) if i != 1))
        if getattr(self, '_optimizer', False):
            self.weights = self._optimizer[0].calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer[1].calculate_update(self.bias, self._gradient_bias)

        return convolution_matrix_full

    def initialize(self, weights_initializer: object, bias_initializer: object) -> None:
        self.weights = weights_initializer.initialize((self.num_kernels, *self.convolution_shape), self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels,), self.fan_in, self.fan_out)

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
