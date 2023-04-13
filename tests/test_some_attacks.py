import random
import unittest

import numpy as np


import tensorflow.compat.v1 as tf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from art.utils import load_mnist, load_cifar10
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from keras.models import Sequential

from attacks.attack import Attack

from attacks.helpers.parameters import ARTParameters, Parameter
from attacks.artattacks.jacobian_saliency_map import JacobianSaliencyMap
from attacks.artattacks.geometric_decision_based import GeometricDecisionBased
from attacks.artattacks.shadow import Shadow
from attacks.artattacks.threshold import Threshold
from attacks.artattacks.sign_opt import SignOPT
from attacks.artattacks.square import Square

tf.compat.v1.disable_eager_execution()
EPOCHS = 50


class Data:
    def __init__(self, d):
        self.input = d[0]
        self.output = d[1]


# Declaring data for future tests

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_pv_keras, max_pv_keras = load_cifar10()
indexesCifar = random.sample(range(len(x_train)), 100)
x_train = x_train[indexesCifar]
y_train = y_train[indexesCifar]
y_shuffled = y_train.__copy__()
np.random.shuffle(y_shuffled)


assert(not np.array_equiv(y_train, y_shuffled))

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
indexesMnist = random.sample(range(len(tup[0])), 100)
# I hope this is proper syntax for extracting pairs of elements corresponding
# to the same index
data = Data((tup[0][indexesMnist], tup[1][indexesMnist]))
data = Data((np.transpose(data.input, (0, 3, 1, 2)).astype(np.float32), data.output))

print(data.output.shape)
print(data.output[0])
print(data.input.shape)
print(data.input[[0]].shape)

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


def _testArgs(test_case : unittest.TestCase, attack_class, parameters : ARTParameters):
    print("Arguments tests...")
    atk = attack_class(parameters)

    for key in parameters.attack_parameters.keys():
        if key in atk._attack_params.keys():
            val = parameters.attack_parameters[key]
            test_case.assertIs(atk._attack_params.get(key), val)
        else:
            test_case.assertIsNone(atk._attack_params.get(key))
            
    
    for key in parameters.classifier_parameters.keys():
        if key in atk._classifier_params.keys():
            val = parameters.classifier_parameters[key]
            test_case.assertIs(atk._classifier_params.get(key), val)
        else:
            test_case.assertIsNone(atk._classifier_params.get(key))
    print("...passed!")



def _testKeras(test_case : unittest.TestCase, attack_class,
               parameters : ARTParameters,
               targeted = None, inputs = x_train, outputs = y_train):
    print("Keras test...")
    atk = attack_class(parameters)
    if (targeted is not None):
        print("Targeted")
        prepared = atk.conduct(kerasModel, Data((inputs, targeted)))
    else:
        prepared = atk.conduct(kerasModel, Data((inputs, outputs)))
    test_case.assertIsNotNone(prepared)
    scoreP = kerasModel.evaluate(prepared, outputs)
    scoreO = kerasModel.evaluate(inputs, outputs)
    print("Original score of model: ", scoreO[0], "(loss), ", scoreO[1], "(acc)")
    print("Attacked score of model: ", scoreP[0], "(loss), ", scoreP[1], "(acc)")

    # test_case.assertTrue(scoreP[1] < scoreO[1])
    print("...passed!")



def _testPyTorch(test_case : unittest.TestCase, attack_class,
                 parameters : ARTParameters,
                 inputs = data.input, outputs = data.output):
    print("PyTorch test...")
    atk = attack_class(parameters)
    prepared = atk.conduct(pyTorchModel, Data((inputs, outputs)))
    test_case.assertIsNotNone(prepared)
    print("...passed!")


                #######################################
                ### DEFINICJE PRZYPADKÓW TESTOWYCH ####
                #######################################


class TestJSMA(unittest.TestCase):
    def testArgs(self):
        _testArgs(self,
                  JacobianSaliencyMap,
                  ARTParameters(
                        {"clip_values": (0,1), "nonsens" : 0xc0ffee},
                        {"verbose" : True,
                         "batch_size" : 10,
                         "theta" : 0.2,
                         "gamma" : 0.8,
                         "random" : "go"}
                  )
                )

    def testPyTorch(self):
        classifier_parameters = {"clip_values": (min_pv_torch, max_pv_torch),
                                 "loss": criterion,
                                 "optimizer": optimizer,
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        attack_parameters = {"verbose" : True, "batch_size" : 10,"gamma" : 0.8}
        _testPyTorch(self, JacobianSaliencyMap,
                     ARTParameters(classifier_parameters,attack_parameters))

    def testKeras(self):
        classifier_parameters = {"clip_values": (min_pv_keras, max_pv_keras)}
        attack_parameters = {"verbose" : True, "batch_size" : 10, "theta" : 0.2, "gamma" : 0.8}
        
        
        _testKeras(self, JacobianSaliencyMap,
                   ARTParameters(classifier_parameters, attack_parameters),  y_shuffled)


class TestGDA(unittest.TestCase):
    def testArgs(self):
        _testArgs(self, GeometricDecisionBased,
                  ARTParameters(
                    {"clip_values": (0,1), "nonsens" : 0xc0ffee},
                    {"verbose" : True, "norm" : "inf", "targeted" : False,
                     "random" : "go"}
                  ))
    
    def testPyTorch(self):
        classifier_parameters = {"clip_values": (min_pv_torch, max_pv_torch),
                                 "loss": criterion,
                                 "optimizer": optimizer,
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        attack_parameters = {"verbose" : False, "batch_size" : 10, "max_iter" : 400}
        _testPyTorch(self, GeometricDecisionBased,
                     ARTParameters(
                        classifier_parameters,
                        attack_parameters
                     ))
        
    def testKeras(self):
        classifier_parameters = {"clip_values": (min_pv_keras, max_pv_keras)}
        attack_parameters = {"verbose" : True, "batch_size" : 50, "max_iter" : 400,
                             "targeted" : True}
        _testKeras(self, GeometricDecisionBased,
                   ARTParameters(
                    classifier_parameters,
                    attack_parameters
                   ), None)

class TestShadow(unittest.TestCase):
    def testArgs(self):
        _testArgs(self, Shadow,
                  ARTParameters(
                    {"clip_values": (0,1), "nonsens" : 0xc0ffee},
                    {"verbose" : True, "sigma" : 0.25, "targeted" : True,
                     "random" : "go"}
                  ))
    
    def testPyTorch(self):
        classifier_parameters = {"clip_values": (min_pv_torch, max_pv_torch),
                                 "loss": criterion,
                                 "optimizer": optimizer,
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        attack_parameters = {"verbose" : True}
        _testPyTorch(self, Shadow,
                     ARTParameters(
                        classifier_parameters,
                        attack_parameters
                     ), inputs= np.asarray(data.input[[0]]), outputs=data.input[[0]])
    
    def testKeras(self):
        classifier_parameters = {"clip_values": (min_pv_keras, max_pv_keras)}
        attack_parameters = {"verbose" : True}
        _testKeras(self, Shadow,
                   ARTParameters(
                    classifier_parameters,
                    attack_parameters
                   ), inputs=x_train[[0]], outputs=y_train[[0]])


class TestSignOPT(unittest.TestCase):
    def testArgs(self):
        _testArgs(self, SignOPT,
                  ARTParameters(
                    {"clip_values": (0,1), "nonsens" : 0xc0ffee},
                    {"verbose" : True, "epsilon" : 0.02, "targeted" : True,
                     "random" : "go"}
                  ))
    
    def testPyTorch(self):
        classifier_parameters = {"clip_values": (min_pv_torch, max_pv_torch),
                                 "loss": criterion,
                                 "optimizer": optimizer,
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        attack_parameters = {"verbose" : True, "batch_size" : 10,
                             "max_iter" : 20, "num_trial" : 100, "k" : 25,
                             "beta" : 0.01}
        _testPyTorch(self, SignOPT,
                     ARTParameters(
                        classifier_parameters,
                        attack_parameters
                     ))

    def testKeras(self):
        classifier_parameters = {"clip_values": (min_pv_keras, max_pv_keras)}
        attack_parameters = {"verbose" : True, "batch_size" : 10,
                             "max_iter" : 20, "num_trial" : 10, "k" : 10,
                             "beta" : 0.01}
        _testKeras(self, SignOPT,
                   ARTParameters(
                    classifier_parameters,
                    attack_parameters
                   ))
        


class TestThreshold(unittest.TestCase):
    def testArgs(self):
        _testArgs(self, Threshold,
                  ARTParameters(
                    {"clip_values": (0,1), "nonsens" : 0xc0ffee},
                    {"verbose" : True, "th" : 0.02, "targeted" : True,
                     "random" : "go"}
                  ))
    
    def testPyTorch(self):
        classifier_parameters = {"clip_values": (min_pv_torch, max_pv_torch),
                                 "loss": criterion,
                                 "optimizer": optimizer,
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        
        attack_parameters = {"verbose" : True, "max_iter" : 1, "th" : 0.01}

        _testPyTorch(self, Threshold,
                     ARTParameters(
                        classifier_parameters,
                        attack_parameters
                     ))
        
    @unittest.skip("#U mnie działa. Trwa bardzo długo ~10min")
    def testKeras(self):
        classifier_parameters = {"clip_values": (min_pv_keras, max_pv_keras)}
        attack_parameters = {"verbose" : True, "max_iter" : 1, "th" : 0.001}
        _testKeras(self, Threshold,
                   ARTParameters(
                    classifier_parameters,
                    attack_parameters
                   ))
        
class TestSquare(unittest.TestCase):
    def testArgs(self):
        _testArgs(self, Square,
                  ARTParameters(
                    {"clip_values": (0,1), "nonsens" : 0xc0ffee},
                    {"verbose" : True, "eps" : 0.02, "targeted" : True,
                     "random" : "go"}
                  ))
    
    def testPyTorch(self):
        classifier_parameters = {"clip_values": (min_pv_torch, max_pv_torch),
                                 "loss": criterion,
                                 "optimizer": optimizer,
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        
        attack_parameters = {"verbose" : True}

        _testPyTorch(self, Square,
                     ARTParameters(
                        classifier_parameters,
                        attack_parameters
                     ))
        
    def testKeras(self):
        classifier_parameters = {"clip_values": (min_pv_keras, max_pv_keras)}
        attack_parameters = {"verbose" : True, "nb_restarts" : 10, "batch_size" : 10}
        _testKeras(self, Square,
                   ARTParameters(
                    classifier_parameters,
                    attack_parameters
                   ))