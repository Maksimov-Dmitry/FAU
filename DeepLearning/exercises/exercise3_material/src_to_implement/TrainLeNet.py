from Layers import Helpers
from Models.LeNet import build
import NeuralNetwork
import matplotlib.pyplot as plt
import os.path
import numpy as np


batch_size = 50
mnist = Helpers.MNISTData(batch_size)
# mnist.show_random_training_image()

if os.path.isfile(os.path.join('trained', 'LeNet')):
    net = NeuralNetwork.load(os.path.join('trained', 'LeNet'), mnist)
else:
    net = build()
    net.data_layer = mnist
    net.train(300)
    NeuralNetwork.save(os.path.join('trained', 'LeNet'), net)

# net.train(300)
# NeuralNetwork.save(os.path.join('trained', 'LeNet'), net)

data, labels = net.data_layer.get_test_set()

results = net.test(data)

accuracy = Helpers.calculate_accuracy(results, labels)
print('\nOn the MNIST dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')
plt.figure('Loss function for training LeNet on the MNIST dataset')
plt.plot(net.loss, '-x')
plt.show()

for i, image in enumerate(data[:10]):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(np.argmax(results[i]))
    plt.show()
