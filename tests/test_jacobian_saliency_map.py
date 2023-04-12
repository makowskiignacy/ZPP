import unittest

import numpy as np
import tensorflow.compat.v1 as tf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from art.utils import load_mnist, load_dataset
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from keras.models import Sequential

from attacks.helpers.parameters import ARTParameters
from attacks.artattacks.jacobian_saliency_map import JacobianSaliencyMap


class Data:
    def __init__(self, d):
        self.input = d[0]
        self.output = d[1]


# Declaring data for future tests
tup, _, min_pixel_value, max_pixel_value = load_mnist()
data = Data(tup)

class TestJSMA(unittest.TestCase):

    def testKeras(self):
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
                                
        attack_parameters = {"verbose" : True, "something_Random" : 21.37}
        parameters = ARTParameters(classifier_parameters, attack_parameters)

        art_attack = JacobianSaliencyMap(parameters)

        self.assertIsNotNone(art_attack.conduct(model, data))


    def testPyTorch(self):
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
        attack_parameters = {"verbose" : True}
        parameters=ARTParameters(classifier_parameters,attack_parameters)

        art_model = JacobianSaliencyMap(parameters)

        # This attack does not need y/output
        data2 = Data((np.transpose(data.input, (0, 3, 1, 2)).astype(np.float32), None))

        self.assertIsNotNone(art_model.conduct(model, data2))

    def test_example(self):
        # Read CIFAR10 dataset
        (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10"))

        x_train, y_train = x_train[:5000], y_train[:5000]
        x_test, y_test = x_test[:500], y_test[:500]

        # Create Keras convolutional neural network - basic architecture from Keras examples
        # Source here: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation("softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        classifier_parameters = {"clip_values": (min_pixel_value, max_pixel_value)}
        attack_parameters = {"verbose" : True}
        parameters = ARTParameters(classifier_parameters, attack_parameters)


        self.assertIsNotNone(JacobianSaliencyMap(parameters).conduct(model, Data((x_test, y_test))))    