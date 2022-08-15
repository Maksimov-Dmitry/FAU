import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):

    def __init__(self, stride_shape: tuple[int, int], pooling_shape: tuple[int, int]):
        super(Pooling, self).__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        pooling_layer_shape = (input_tensor.shape[0], 
                     input_tensor.shape[1], 
                     int(1 + (input_tensor.shape[2] - self.pooling_shape[0]) / self.stride_shape[0]),
                     int(1 + (input_tensor.shape[3] - self.pooling_shape[1]) / self.stride_shape[1])
        )
        pooling_tenzor = np.zeros(pooling_layer_shape)
        self.pooling_tenzor_argmax = np.zeros((*pooling_layer_shape, 2), dtype=int)
        self.input_tensor_shape = input_tensor.shape
        for image_index, image in enumerate(input_tensor):
            for channel_index, channel in enumerate(image):
                for heigh_index in range(pooling_layer_shape[2]):
                    for width_index in range(pooling_layer_shape[3]):
                        vert_start = heigh_index * self.stride_shape[0]
                        vert_end = vert_start + self.pooling_shape[0]
                        horiz_start = width_index * self.stride_shape[1]
                        horiz_end = horiz_start + self.pooling_shape[1]

                        channel_slice = channel[vert_start:vert_end, horiz_start:horiz_end]
                        pooling_tenzor[image_index, channel_index, heigh_index, width_index] = np.max(channel_slice)
                        argmax = np.unravel_index(np.argmax(channel_slice), channel_slice.shape)
                        argmax_full = np.array(tuple(map(sum, zip(argmax, (vert_start, horiz_start)))))
                        self.pooling_tenzor_argmax[image_index, channel_index, heigh_index, width_index] = argmax_full
        return pooling_tenzor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        downstream_gradient = np.zeros(self.input_tensor_shape)
        for image_index, image in enumerate(error_tensor):
            for channel_index, channel in enumerate(image):
                for high_index, high in enumerate(channel):
                    for width_index, max in enumerate(high):
                        argmax_index = (image_index, channel_index, *self.pooling_tenzor_argmax[image_index, channel_index, high_index, width_index])
                        downstream_gradient[argmax_index] += max
        return downstream_gradient