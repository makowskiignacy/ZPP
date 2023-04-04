import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from art.utils import load_mnist


from attacks.helpers.parameters import ARTParameters
from attacks.artattacks.zeroth_order_optimization_bb_attack import ZeorthOrderOptimalization


_mnist_data, _, _min_pixel_value, _max_pixel_value = load_mnist()

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

    def convert_tensors_to_numpy(self):
        if isinstance(self.input, torch.Tensor):
            self.input = self.input.numpy

        if isinstance(self.output, torch.Tensor):
            self.input = self.output.numpy

        return self

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


class ZooTestUsingPytorchArt(unittest.TestCase):
    def test_art_ZeorthOrderOptimalization(self):
        model = pytorch_model_form_art()
        # This is only for testing purpose as are the parameters
        model.cpu()
        classifier_parameters = {"clip_values": (_min_pixel_value, _max_pixel_value),
                                 "loss": nn.CrossEntropyLoss(),
                                 "optimizer": optim.Adam(model.parameters(), lr=0.01),
                                 "input_shape": (1, 28, 28),
                                 "nb_classes": 10}
        attack_parameters = {"nb_parallel": 1,
                             "binary_search_steps": 1,
                             "batch_size": 1000,
                             "max_iter": 1,
                             "verbose": True}
        parameters = ARTParameters(classifier_parameters=classifier_parameters, attack_parameters=attack_parameters)

        art_model = ZeorthOrderOptimalization(parameters=parameters)

        self.assertIsNotNone(art_model.conduct(pytorch_model_form_art(), art_sample_data()))

class ZooTestUsingPytorchArt__potential_failure(unittest.TestCase):
    def test_art_ZeorthOrderOptimalization(self):
        model = pytorch_model_form_art()
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
                             "max_iter": 1,
                             "use_resize": False,
                             "verbose": True}
        parameters = ARTParameters(classifier_parameters=classifier_parameters, attack_parameters=attack_parameters)

        art_model = ZeorthOrderOptimalization(parameters=parameters)

        self.assertIsNotNone(art_model.conduct(pytorch_model_form_art(), art_sample_data()))