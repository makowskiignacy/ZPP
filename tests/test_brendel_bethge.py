import time
import mlflow
import csv
import numpy as np
import torch
import unittest

import torchvision.models as tv_models
import foolbox as fb
from attacks.foolboxattacks.brendel_bethge import L0BrendelBethge, L1BrendelBethge, L2BrendelBethge, LinfinityBrendelBethge
from attacks.helpers.data import Data
from attacks.helpers.parameters import FoolboxParameters
from attacks.foolbox_attack import FoolboxAttack
from foolbox.utils import accuracy

from foolbox.models.pytorch import PyTorchModel

def simple_test():
    model = tv_models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fb.models.pytorch.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=4)
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


def conduct(attack: FoolboxAttack, model, data: Data):
        time_start = time.time()

        # print(type(model))
        if isinstance(model, PyTorchModel) :
            print(f"Model accuracy before attack: {fb.utils.accuracy(model, data.input, data.output)}")
        print(f"Starting attack. ({time.asctime(time.localtime(time_start))})")

        adversarials = attack.conduct(model, data)

        time_end = time.time()
        print(f"Attack done. ({time.asctime(time.localtime(time_end))})")
        print(f"Took {time_end - time_start}\n")

        if adversarials is not None and isinstance(model, PyTorchModel):
            print(f"Model accuracy after attack: {accuracy(model, adversarials, data.output)}")

        return adversarials


        
class TestBrendelBethge(unittest.TestCase):
    generic_parameters_simple = {'min': 0, 'max': 1}
    generic_parameters_nn = {'min': -1.0, 'max': 28157.0}
    attack_specific_parameters = {"lr": 10, 'steps': 100}
    parameters_simple = FoolboxParameters(attack_specific_parameters,generic_parameters_simple)
    parameters_nn = FoolboxParameters(attack_specific_parameters, generic_parameters_nn)

    attack_bb0_simple = L0BrendelBethge(parameters_simple)
    attack_bb1_simple = L1BrendelBethge(parameters_simple)
    attack_bb2_simple = L2BrendelBethge(parameters_simple)
    attack_bbinf_simple = LinfinityBrendelBethge(parameters_simple)

    attack_bb0_nn = L0BrendelBethge(parameters_nn)
    attack_bb1_nn = L1BrendelBethge(parameters_nn)
    attack_bb2_nn = L2BrendelBethge(parameters_nn)
    attack_bbinf_nn = LinfinityBrendelBethge(parameters_nn)

    smodel, sdata = simple_test()

    def test_bb_0_simple(self):
        result0s = conduct(self.attack_bb0_simple, self.smodel, self.sdata)
        #print(result0s)

    def test_bb_1_simple(self):
        result1s = conduct(self.attack_bb1_simple, self.smodel, self.sdata)
        #print(result1s)

    def test_bb_2_simple(self):
        result2s = conduct(self.attack_bb2_simple, self.smodel, self.sdata)
        #print(result2s)

    def test_bb_inf_simple(self):
        resultinfs = conduct(self.attack_bbinf_simple, self.smodel, self.sdata)
        #print(resultinfs)

    nn_model, nn_data = nn_test()

    def test_bb_0_nn(self):
        result0nn = conduct(self.attack_bb0_nn, self.nn_model, self.nn_data)
        #print(result0nn)

    def test_bb_1_nn(self):
        result1nn = conduct(self.attack_bb1_nn, self.nn_model, self.nn_data)
        #print(result1nn)

    def test_bb_2_nn(self):
        result2nn = conduct(self.attack_bb2_nn, self.nn_model, self.nn_data)
        #print(result2nn)

    def test_bb_inf_nn(self):
        resultinfnn = conduct(self.attack_bbinf_nn, self.nn_model, self.nn_data)
        #print(resultinfnn)

