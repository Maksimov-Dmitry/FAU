import numpy as np
from typing import Tuple

class Constant:

    def __init__(self, const: float = 0.1):
        self.const = const

    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int):
        return np.full(weights_shape, self.const)


class UniformRandom:

    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int):
        return np.random.uniform(size=weights_shape)


class Xavier:

    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int):
        sigma = np.sqrt(2/(fan_in + fan_out))
        return np.random.normal(scale=sigma, size=weights_shape)


class He:

    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int):
        sigma = np.sqrt(2/fan_in)
        return np.random.normal(scale=sigma, size=weights_shape)
