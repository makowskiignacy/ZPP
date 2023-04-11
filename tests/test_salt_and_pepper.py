from attacks.foolboxattacks.salt_and_pepper import SaltAndPepperNoise

import time
import mlflow
import csv
import numpy as np
import torch
import unittest

import torchvision.models as tv_models
import foolbox as fb
from attacks.helpers.data import Data
from attacks.helpers.parameters import FoolboxParameters
from attacks.foolbox_attack import FoolboxAttack
from foolbox.utils import accuracy

from foolbox.models.pytorch import PyTorchModel


def simple_test(batchsize=40):
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fb.models.pytorch.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=batchsize)
    data = Data(images, labels)

    return fmodel, data


def nn_test():
    ss_nn_pipeline = mlflow.sklearn.load_model('../ss_nn/')
    if ss_nn_pipeline is not None:
        standard_scaler_from_nn_pipeline = ss_nn_pipeline.steps[0][1]
        nn_model = ss_nn_pipeline.steps[1][1].module_
        fmodel = fb.models.pytorch.PyTorchModel(nn_model, bounds=(-2., 30000.))

        csv_filename = '../data_test.csv'

        with open(csv_filename) as f:
            reader = csv.reader(f)
            data = list(list(line) for line in reader)
            data.pop(0)
            data2 = []
            i = 0
            for row in data:
                if i < 100:
                    i = i+1
                    row2 = []
                    for place in range(len(row)):
                        row2.append(float(row[place]))
                    data2.append(row2)
            data = data2
            data = torch.tensor(data, requires_grad=False, dtype=torch.float)
            data, result = torch.hsplit(data, [91, ])
            result = torch.tensor(result, requires_grad=False, dtype=torch.float)
            data = Data(data, result)

        return fmodel, data
    return None, None


def conduct(attack: SaltAndPepperNoise, model, data: Data):
    time_start = time.time()

    # print(type(model))
    if isinstance(model, PyTorchModel):
        print(f"Model accuracy before attack: {fb.utils.accuracy(model, data.input, data.output)}")
    print(f"Starting attack. ({time.asctime(time.localtime(time_start))})")

    adversarials = attack.conduct(model, data)

    time_end = time.time()
    print(f"Attack done. ({time.asctime(time.localtime(time_end))})")
    print(f"Took {time_end - time_start}\n")

    if adversarials is not None and isinstance(model, PyTorchModel):
        print(f"Model accuracy after attack: {accuracy(model, adversarials, data.output)}")

    return adversarials


class TestSaltAndPepper(unittest.TestCase):
    attack_specific_parameters_simple = {"steps": 10, "across_channels": True}
    attack_specific_parameters_nn = {"steps": 100, "across_channels": True}
    generic_parameters_simple = {}
    generic_parameters_nn = {}
    parameters_simple = FoolboxParameters(attack_specific_parameters_simple, generic_parameters_simple)
    parameters_nn = FoolboxParameters(attack_specific_parameters_nn, generic_parameters_nn)


    attack_sap_simple = SaltAndPepperNoise(parameters_simple)
    attack_sap_nn = SaltAndPepperNoise(parameters_nn)

    def test_sap_simple_smaller(self):
        smodel, sdata = simple_test(batchsize=4)
        result1s = conduct(self.attack_sap_simple, smodel, sdata)
        self.assertIsNotNone(result1s)

    def test_sap_simple_larger(self):
        smodel, sdata = simple_test(batchsize=20)
        result2s = conduct(self.attack_sap_simple, smodel, sdata)
        self.assertIsNotNone(result2s)

    def test_sap_nn(self):
        nn_model, nn_data = nn_test()
        resultnn = conduct(self.attack_sap_nn, nn_model, nn_data)
        self.assertIsNotNone(resultnn)


