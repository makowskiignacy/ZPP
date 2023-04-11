import time
import mlflow
import csv
import numpy as np
import torch
import unittest

import torchvision.models as tv_models
import foolbox as fb
from attacks.foolboxattacks.basic_iterative import L1BasicIterative, L2BasicIterative, LinfBasicIterative
from attacks.foolboxattacks.basic_iterative import L1AdamBasicIterative, L2AdamBasicIterative, LinfAdamBasicIterative
from attacks.helpers.parameters import FoolboxParameters
from attacks.helpers.data import Data
from attacks.foolbox_attack import FoolboxAttack
from foolbox.utils import accuracy
from foolbox.models.pytorch import PyTorchModel



def simple_test(batchsize=4):
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

        csv_filename = '../data_test.csv'

        with open(csv_filename) as f:
            reader = csv.reader(f)
            data = list(list(line) for line in reader)
            data.pop(0)
            data2 = []
            for row in data:
                row2 = []
                for place in range(len(row)):
                    row2.append(float(row[place]))
                data2.append(row2)
            data = data2
            data = torch.tensor(data, requires_grad=False, dtype=torch.float)
            data, result = torch.hsplit(data, [91, ])
            result = torch.tensor(result, requires_grad=False, dtype=torch.float)
            data = Data(data, result)

        return nn_model, data
    return None, None


def conduct(attack: FoolboxAttack, model, data: Data):
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


class TestBasicIterative(unittest.TestCase):
    attack_specific_parameters = {"steps": 10, "random_start": True}
    generic_parameters = {"epsilon_rate": 0.01}
    parameters = FoolboxParameters(attack_specific_parameters, generic_parameters)

    attack_bi1 = L1BasicIterative(parameters)
    attack_bi2 = L2BasicIterative(parameters)
    attack_biinf = LinfBasicIterative(parameters)
    attack_bi1_a = L1AdamBasicIterative(parameters)
    attack_bi2_a = L2AdamBasicIterative(parameters)
    attack_biinf_a = LinfAdamBasicIterative(parameters)

    smodel, sdata = simple_test(batchsize=20)

    def test_bi_1_simple(self):
        result1s = conduct(self.attack_bi1, self.smodel, self.sdata)
        self.assertIsNotNone(result1s)

    def test_bi_2_simple(self):
        result2s = conduct(self.attack_bi2, self.smodel, self.sdata)
        self.assertIsNotNone(result2s)

    def test_bi_inf_simple(self):
        resultinfs = conduct(self.attack_biinf, self.smodel, self.sdata)
        self.assertIsNotNone(resultinfs)

    def test_bi_1_a_simple(self):
        result1s = conduct(self.attack_bi1_a, self.smodel, self.sdata)
        self.assertIsNotNone(result1s)

    def test_bi_2_a_simple(self):
        result2s = conduct(self.attack_bi2_a, self.smodel, self.sdata)
        self.assertIsNotNone(result2s)

    def test_bi_inf_a_simple(self):
        resultinfs = conduct(self.attack_biinf_a, self.smodel, self.sdata)
        self.assertIsNotNone(resultinfs)

    nn_model, nn_data = nn_test()

    def test_bi_1_nn(self):
        result1nn = conduct(self.attack_bi1, self.nn_model, self.nn_data)
        self.assertIsNotNone(result1nn)

    def test_bi_2_nn(self):
        result2nn = conduct(self.attack_bi2, self.nn_model, self.nn_data)
        self.assertIsNotNone(result2nn)

    def test_bi_inf_nn(self):
        resultinfnn = conduct(self.attack_biinf, self.nn_model, self.nn_data)
        self.assertIsNotNone(resultinfnn)

    def test_bi_1_a_nn(self):
        result1ann = conduct(self.attack_bi1_a, self.nn_model, self.nn_data)
        self.assertIsNotNone(result1ann)

    def test_bi_2_a_nn(self):
        result2ann = conduct(self.attack_bi2_a, self.nn_model, self.nn_data)
        self.assertIsNotNone(result2ann)

    def test_bi_inf_a_nn(self):
        resultinfann = conduct(self.attack_biinf_a, self.nn_model, self.nn_data)
        self.assertIsNotNone(resultinfann)

