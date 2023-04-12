import random
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

tf.compat.v1.disable_eager_execution()
EPOCHS = 50


class Data:
    def __init__(self, d):
        self.input = d[0]
        self.output = d[1]


# Declaring data for future tests

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_pv_keras, max_pv_keras = load_dataset(str("cifar10"))
indexesCifar = random.sample(range(len(x_train)), 200)
x_train = x_train[indexesCifar]
y_train = y_train[indexesCifar]

# Create Keras convolutional neural network - basic architecture from Keras examples
# Source here: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
kerasModel = Sequential()
kerasModel.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
kerasModel.add(Activation("relu"))
kerasModel.add(Conv2D(32, (3, 3)))
kerasModel.add(Activation("relu"))
kerasModel.add(MaxPooling2D(pool_size=(2, 2)))
kerasModel.add(Dropout(0.25))

kerasModel.add(Conv2D(64, (3, 3), padding="same"))
kerasModel.add(Activation("relu"))
kerasModel.add(Conv2D(64, (3, 3)))
kerasModel.add(Activation("relu"))
kerasModel.add(MaxPooling2D(pool_size=(2, 2)))
kerasModel.add(Dropout(0.25))

kerasModel.add(Flatten())
kerasModel.add(Dense(512))
kerasModel.add(Activation("relu"))
kerasModel.add(Dropout(0.5))
kerasModel.add(Dense(10))
kerasModel.add(Activation("softmax"))

kerasModel.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
kerasModel.fit(x_train, y_train, epochs=EPOCHS, batch_size=len(x_train)//10)



tup, _, min_pv_torch, max_pv_torch = load_mnist()
indexesMnist = random.sample(range(len(tup[0])), 200)
# I hope this is proper syntax for extracting pairs of elements corresponding
# to the same index
data = Data((tup[0][indexesMnist], tup[1][indexesMnist]))

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
pyTorchModel = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pyTorchModel.parameters(), lr=0.01)

# Here I get the following error, which I cannot fix easily
# so i commented off the learning part of PyTorch model
# TypeError: conv2d() received an invalid combination of arguments - got (numpy.ndarray, Parameter, Parameter, tuple, tuple, tuple, int), but expected one of:
# * (Tensor input, Tensor weight, Tensor bias, tuple of ints stride, tuple of ints padding, tuple of ints dilation, int groups)
#      didn't match because some of the arguments have invalid types: (numpy.ndarray, Parameter, Parameter, tuple, tuple, tuple, int)
# for e in range(EPOCHS):
#     pyTorchModel.train(True)
#     optimizer.zero_grad()
#     outputs = pyTorchModel(data.input)
#     loss = criterion(outputs, data.output)
#     loss.backward()
#     optimizer.step()

class TestJSMA(unittest.TestCase):
    def testArgs(self):
        print("testParams()") 
        atk = JacobianSaliencyMap(
            ARTParameters(
                {"clip_values": (0,1), "nonsens" : 0xc0ffee},
                {"verbose" : True, "random" : "go"}
                )
        )
        self.assertIs(atk._attack_params.get("verbose", False), True)
        self.assertIs(atk._attack_params.get("random", "no"), "no")
        self.assertIs(atk._classifier_params.get("nonsens", 1), 1)
        self.assertIs(atk._classifier_params.get("clip_values", (-1,-1)), (0,1))


    def testPyTorch(self):
        
        classifier_parameters = {"clip_values": (min_pv_torch, max_pv_torch),
                                 "loss": criterion,
                                 "optimizer": optimizer,
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        attack_parameters = {"verbose" : True}
        parameters=ARTParameters(classifier_parameters,attack_parameters)

        attack = JacobianSaliencyMap(parameters)

        # This attack does not need y/output
        data2 = Data((np.transpose(data.input, (0, 3, 1, 2)).astype(np.float32), None))

        self.assertIsNotNone(attack.conduct(pyTorchModel, data2))

    def testKeras(self):
        classifier_parameters = {"clip_values": (min_pv_keras, max_pv_keras)}
        attack_parameters = {"verbose" : True}
        parameters = ARTParameters(classifier_parameters, attack_parameters)

        # We don't need to conduct attack on 'test' set
        self.assertIsNotNone(JacobianSaliencyMap(parameters).conduct(kerasModel, Data((x_train, y_train))))    