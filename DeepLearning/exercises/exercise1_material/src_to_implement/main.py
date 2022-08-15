from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected

if __name__ == '__main__':
    base = BaseLayer()
    fully_connected = FullyConnected(5, 5)
    print(fully_connected.trainable)
