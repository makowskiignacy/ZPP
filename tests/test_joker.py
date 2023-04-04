'''
A set of simple tests to check the compatibility of the FastGradient attack with supported libraries.
'''

import unittest

import numpy as np
import tensorflow.compat.v1 as tf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from art.utils import load_mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

from attacks.artattacks.joker import Joker
from attacks.helpers.parameters import ARTParameters


class Data:
    def __init__(self, d):
        self.input = d[0]
        self.output = d[1]


# Declaring data for future tests
tup, _, min_pixel_value, max_pixel_value = load_mnist()
data = Data(tup)

class TestKeras(unittest.TestCase):
    def test_keras(self):
        tf.compat.v1.disable_eager_execution()

        model = Sequential()
        model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dense(10, activation="softmax"))

        classifier_parameters = {"clip_values": (min_pixel_value, max_pixel_value)}
        attack_parameters = {}
        parameters = ARTParameters(classifier_parameters=classifier_parameters, attack_parameters=attack_parameters)

        art_attack = Joker(joker='FastGradientMethod', parameters=parameters)

        self.assertIsNotNone(art_attack.conduct(model, data))


class TestPytorch(unittest.TestCase):
    def test_pytorch(self):
        # declaring pytorch model
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
                self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
                self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
                self.fc_2 = nn.Linear(in_features=100, out_features=10)

            def forward(self, x):
                x = F.relu(self.conv_1(x))
                x = F.max_pool2d(x, 2, 2)
                x = F.relu(self.conv_2(x))
                x = F.max_pool2d(x, 2, 2)
                x = x.view(-1, 4 * 4 * 10)
                x = F.relu(self.fc_1(x))
                x = self.fc_2(x)
                return x

        # Initializing parameters of the attack
        model = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        classifier_parameters = {"clip_values": (min_pixel_value, max_pixel_value),
                                 "loss": criterion,
                                 "optimizer": optimizer,
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        attack_parameters = {}
        parameters = ARTParameters(classifier_parameters=classifier_parameters, attack_parameters=attack_parameters)

        art_model = Joker(joker='FastGradientMethod', parameters=parameters)

        # This attack does not need y/output
        data2 = Data((np.transpose(data.input, (0, 3, 1, 2)).astype(np.float32), None))
        self.assertIsNotNone(art_model.conduct(model, data2))
