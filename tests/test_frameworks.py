import unittest

import torchvision.models as models
import torch.nn.functional as F
import eagerpy as ep
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

from foolbox import PyTorchModel, samples
from art.utils import load_mnist

from attacks.artattacks.adversarial_patch import AdversarialPatch
from attacks.artattacks.zeroth_order_optimization_bb_attack import ZeorthOrderOptimalization
from attacks.foolboxattacks.projected_gradient_descent import ProjectedGradientDescentInf
from attacks.foolboxattacks.L1_basic_iterative import L1BasicIterative

import timeout_decorator


# Wrapper class providing .input and .output for some of the attacks
class Data:
    def __init__(self, data):
        self.input = data[0]
        self.output = data[1]

    def foolbox_pytorch_preprocessing(self):
        # Labels are expected to be 1D, match the length of logits (not checked) and type Long
        # It is convenient for vectors to be converted to tensors, it bypass NotImplementedError trowed for np
        self.input = torch.from_numpy(np.array(self.input))
        self.output = torch.from_numpy(self.output.astype(np.longlong))
        if self.output.ndim > 1:
            self.output = self.output[:,0]
        return self


# Declaring global variables for
_model = models.resnet18(pretrained=True).eval()
_fmodel = PyTorchModel(_model, bounds=(0, 1))
_data = ep.astensors(*samples(_fmodel, dataset="imagenet", batchsize=16))
_mnist_data, _, _min_pixel_value, _max_pixel_value = load_mnist()


def pytorch_model_form_foolbox():
    return _model


def foolbox_sample_data():
    return Data(_data)


def pytorch_model_form_art():
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


def art_sample_data():
    return Data((np.transpose(_mnist_data[0], (0, 3, 1, 2)).astype(np.float32), _mnist_data[1]))


class TestFoolboxWithFoolboxExamples(unittest.TestCase):
    def test_foolbox_ProjectedGradientDescentInf(self):
        foolbox_model = ProjectedGradientDescentInf()
        self.assertIsNotNone(foolbox_model.conduct(pytorch_model_form_foolbox(), foolbox_sample_data()))

    def test_foolbox_L1BasicIterative(self):
        foolbox_model = L1BasicIterative({})
        self.assertIsNotNone(foolbox_model.conduct(pytorch_model_form_foolbox(), foolbox_sample_data()))


class TestFoolboxWithArtExamples(unittest.TestCase):
    def test_foolbox_ProjectedGradientDescentInf(self):
        foolbox_model = ProjectedGradientDescentInf()
        self.assertIsNotNone(foolbox_model.conduct(pytorch_model_form_art(),
                                                   art_sample_data().foolbox_pytorch_preprocessing()))

    def test_foolbox_L1BasicIterative(self):
        foolbox_model = L1BasicIterative({})
        self.assertIsNotNone(foolbox_model.conduct(pytorch_model_form_art(),
                                                   art_sample_data().foolbox_pytorch_preprocessing()))


class TestArtWithArtExamples(unittest.TestCase):
    @timeout_decorator.timeout(60)
    def test_art_ZeorthOrderOptimalization(self):
        model = pytorch_model_form_art()
        art_model = ZeorthOrderOptimalization(clip_values=(_min_pixel_value, _max_pixel_value),
                                              loss=nn.CrossEntropyLoss(),
                                              optimizer=optim.Adam(model.parameters(), lr=0.01),
                                              input_shape=(1, 28, 28),
                                              nb_classes=10)
        self.assertIsNotNone(art_model.conduct(pytorch_model_form_art(), art_sample_data()))

    @timeout_decorator.timeout(60)
    def test_art_AdversarialPatch(self):
        art_model = AdversarialPatch({})
        self.assertIsNotNone(art_model.conduct(pytorch_model_form_art(), art_sample_data()))


class TestArtWithFoolboxExamples(unittest.TestCase):
    @timeout_decorator.timeout(60)
    def test_art_ZeorthOrderOptimalization(self):
        model = pytorch_model_form_art()
        art_model = ZeorthOrderOptimalization(clip_values=(_min_pixel_value, _max_pixel_value),
                                              loss=nn.CrossEntropyLoss(),
                                              optimizer=optim.Adam(model.parameters(), lr=0.01),
                                              input_shape=(1, 28, 28),
                                              nb_classes=10)
        self.assertIsNotNone(art_model.conduct(pytorch_model_form_foolbox(), foolbox_sample_data()))

    @timeout_decorator.timeout(60)
    def test_art_AdversarialPatch(self):
        art_model = AdversarialPatch({})
        self.assertIsNotNone(art_model.conduct(pytorch_model_form_foolbox(), foolbox_sample_data()))


if __name__ == '__main__':
    unittest.main()