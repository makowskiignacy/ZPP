'''
The file contains simple tests to check the compatibility of attacks with different frameworks and data types.

It is based on sample models and data posted on foolbox and arta repositories

The tests are not intended to check the correctness of the results but only whether the attacks will run correctly
'''

import unittest

import eagerpy as ep
import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from art.utils import load_mnist
from foolbox import PyTorchModel, samples
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam

from attacks.artattacks.adversarial_patch import AdversarialPatch
from attacks.artattacks.fast_gradient import FastGradient
from attacks.artattacks.zeroth_order_optimization_bb_attack import ZerothOrderOptimalization
from attacks.foolboxattacks.basic_iterative import L1BasicIterative, L2BasicIterative, LinfBasicIterative
from attacks.foolboxattacks.projected_gradient_descent import LinfProjectedGradientDescent
from attacks.helpers.parameters import FoolboxParameters, ARTParameters

tf.compat.v1.disable_eager_execution()


# Wrapper class providing .input and .output for some of the attacks
class Data:
    def __init__(self, x, y):
        self.input = x
        self.output = y

    def foolbox_pytorch_preprocessing(self):
        # Labels are expected to be 1D, match the length of logits (not checked) and type Long
        # It is convenient for vectors to be converted to tensors, it bypass NotImplementedError trowed for np
        self.input = torch.from_numpy(np.array(self.input))
        self.output = torch.from_numpy(self.output.astype(np.longlong))
        if self.output.ndim > 1:
            self.output = self.output[:,0]
        return self

    def convert_tensors_to_numpy(self):
        if isinstance(self.input, torch.Tensor):
            self.input = self.input.numpy

        if isinstance(self.output, torch.Tensor):
            self.output = self.output.numpy

        return self


# Declaring global variables for
_foolbox_model = models.resnet18(pretrained=True).eval()
_fmodel = PyTorchModel(_foolbox_model, bounds=(0, 1))
_foolbox_data = ep.astensors(*samples(_fmodel, dataset="imagenet", batchsize=16))
_mnist_data, _, _min_pixel_value, _max_pixel_value = load_mnist()

_keras_art_model = Sequential()
_keras_art_model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
_keras_art_model.add(MaxPooling2D(pool_size=(2, 2)))
_keras_art_model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)))
_keras_art_model.add(MaxPooling2D(pool_size=(2, 2)))
_keras_art_model.add(Flatten())
_keras_art_model.add(Dense(100, activation="relu"))
_keras_art_model.add(Dense(10, activation="softmax"))

_keras_art_model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

_keras_foolbox_model = tensorflow.keras.applications.ResNet50(weights="imagenet")


def foolbox_sample_data():
    return Data(_foolbox_data[0], _foolbox_data[1])


def art_sample_data():
    return Data(np.transpose(_mnist_data[0], (0, 3, 1, 2)).astype(np.float32), _mnist_data[1])



def pytorch_model_from_foolbox():
    return _foolbox_model


def pytorch_model_from_art():
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
    return Net().eval()


def keras_model_from_art():
    return _keras_art_model


def keras_model_from_foolbox():
    return _keras_foolbox_model


class TestFoolboxWithPytorchUsingFoolbox:
    '''
    Checks attacks from foolboxattacks package using example from:
    https://github.com/bethgelab/foolbox/blob/master/examples/single_attack_pytorch_resnet18.py
    '''
    def test_foolbox_ProjectedGradientDescentInf(self):
        foolbox_model = LinfProjectedGradientDescent(FoolboxParameters({}, {}))
        return foolbox_model.conduct(pytorch_model_from_foolbox(), foolbox_sample_data())

    def test_foolbox_L1BasicIterative(self):
        foolbox_model = L1BasicIterative(FoolboxParameters({}, {}))
        return foolbox_model.conduct(pytorch_model_from_foolbox(), foolbox_sample_data())


# Pytorch test
class TestFoolboxWithPytorchUsingArt:
    '''
    Checks attacks from foolboxattacks package using example from:
    https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/get_started_pytorch.py
    '''
    def test_foolbox_ProjectedGradientDescentInf(self):
        foolbox_model = LinfProjectedGradientDescent(FoolboxParameters({}, {}))
        return foolbox_model.conduct(pytorch_model_from_art(), art_sample_data().foolbox_pytorch_preprocessing())

    def test_foolbox_L1BasicIterative(self):
        foolbox_model = L1BasicIterative(FoolboxParameters({}, {}))
        return foolbox_model.conduct(pytorch_model_from_art(),art_sample_data().foolbox_pytorch_preprocessing())

    def test_foolbox_L2BasicIterative(self):
        foolbox_model = L2BasicIterative(FoolboxParameters({}, {}))
        return foolbox_model.conduct(pytorch_model_from_art(), art_sample_data().foolbox_pytorch_preprocessing())

    def test_foolbox_LinfBasicIterative(self):
        foolbox_model = LinfBasicIterative(FoolboxParameters({}, {}))
        return foolbox_model.conduct(pytorch_model_from_art(), art_sample_data().foolbox_pytorch_preprocessing())


class TestArtWithPytorchUsingArt:
    '''
    Checks attacks from artattacks package using example from:
    https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/get_started_pytorch.py
    '''
    def test_art_ZerothOrderOptimalization(self):
        model = pytorch_model_from_art()
        # This is only for testing purpose as are the parameters
        model.cpu()
        classifier_parameters = {"clip_values": (_min_pixel_value, _max_pixel_value),
                                 "loss": nn.CrossEntropyLoss(),
                                 "optimizer": optim.Adam(model.parameters(), lr=0.01),
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        attack_parameters = {"nb_parallel": 1,
                             "binary_search_steps": 1,
                             "batch_size": 500,
                             "max_iter": 2,
                             "verbose": True}

        parameters = ARTParameters(classifier_parameters=classifier_parameters, attack_parameters=attack_parameters)

        art_model = ZerothOrderOptimalization(parameters=parameters)

        return art_model.conduct(pytorch_model_from_art(), art_sample_data())

    def test_art_AdversarialPatch(self):
        art_model = AdversarialPatch(ARTParameters({}, {}))
        return art_model.conduct(pytorch_model_from_art(), art_sample_data())

    def test_art_FastGradient(self):
        model = pytorch_model_from_art()
        classifier_parameters = {"clip_values": (_min_pixel_value, _max_pixel_value), "use_logits": False}
        attack_parameters = {}
        parameters = ARTParameters(classifier_parameters=classifier_parameters, attack_parameters=attack_parameters)
        art_model = FastGradient(parameters=parameters)
        return art_model.conduct(model, Data(load_mnist()[0][0], load_mnist()[0][1]))


class TestArtWithPytorchUsingFoolbox:
    '''
    Checks attacks from artattacks package using example from:
    https://github.com/bethgelab/foolbox/blob/master/examples/single_attack_pytorch_resnet18.py
    '''
    def test_art_ZerothOrderOptimalization(self):
        model = pytorch_model_from_foolbox()
        classifier_parameters = {"clip_values": (_min_pixel_value, _max_pixel_value),
                                 "loss": nn.CrossEntropyLoss(),
                                 "optimizer": optim.Adam(model.parameters(), lr=0.01),
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        attack_parameters = {"nb_parallel": 1,
                             "binary_search_steps": 1,
                             "batch_size": 500,
                             "max_iter": 2,
                             "verbose": True}

        parameters = ARTParameters(classifier_parameters=classifier_parameters, attack_parameters=attack_parameters)

        art_model = ZerothOrderOptimalization(parameters=parameters)

        # FIXME :
        # Poniższa linijka wyrzuca błąd:
        # File "/mnt/data/Studia/All_ZPP/ZPP/venv/lib/python3.10/site-packages/art/utils.py", line 738, in to_categorical
        # labels = np.array(labels, dtype=int)
        # ValueError: setting an array element with a sequence.
        # Prawdopodobnie coś związanego z reprezentajcą poszczególnych klas
        # Czy Foolbox używa tablic? Czy kodowanie jest hot-one?
        return art_model.conduct(model, foolbox_sample_data())

    def test_art_FastGradient(self):
        model = pytorch_model_from_foolbox()
        classifier_parameters = {"clip_values": (_min_pixel_value, _max_pixel_value), "use_logits": False}
        attack_parameters = {}
        parameters = ARTParameters(classifier_parameters=classifier_parameters, attack_parameters=attack_parameters)
        art_model = FastGradient(parameters=parameters)
        return art_model.conduct(model, Data(load_mnist()[0][0], load_mnist()[0][1]))


    def test_art_AdversarialPatch(self):
        art_model = AdversarialPatch(ARTParameters({}, {}))
        return art_model.conduct(pytorch_model_from_foolbox(), foolbox_sample_data())


class TestArtWithKerasUsingArt:
    def test_art_FastGradient(self):
        mod = keras_model_from_art()

        classifier_parameters = {"clip_values": (_min_pixel_value, _max_pixel_value),
                                 "use_logits": False}
        attack_parameters = {}
        parameters = ARTParameters(classifier_parameters=classifier_parameters, attack_parameters=attack_parameters)
        art_model = FastGradient(parameters=parameters)
        return art_model.conduct(mod, Data(load_mnist()[0][0], load_mnist()[0][1]))

    def test_art_ZerothOrderOptimalization(self):
        model = pytorch_model_from_art()
        classifier_parameters = {"clip_values": (_min_pixel_value, _max_pixel_value),
                                 "loss": nn.CrossEntropyLoss(),
                                 "optimizer": optim.Adam(model.parameters(), lr=0.01),
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        attack_parameters = {}

        parameters = ARTParameters(classifier_parameters=classifier_parameters, attack_parameters=attack_parameters)

        art_model = ZerothOrderOptimalization(parameters=parameters)

        return art_model.conduct(keras_model_from_art(), Data(load_mnist()[0][0], load_mnist()[0][1]))

    def test_art_AdversarialPatch(self):
        art_model = AdversarialPatch(ARTParameters({}, {}))
        return art_model.conduct(keras_model_from_art(), Data(load_mnist()[0][0], load_mnist()[0][1]))
