from abc import ABC, abstractmethod
from attacks.attack import Attack

from foolbox.models.pytorch import PyTorchModel
import torchvision.models as tv_models
import mlflow
import foolbox as fb
import torch
from attacks.helpers.data import Data

import time
import csv
import numpy as np

from attacks.foolbox_attack import FoolboxAttack
from foolbox.utils import accuracy

import pandas as pd
import os

CWD = os.getcwd()
SS_NN_RELATIVE_PATH = 'ss_nn/'
DATA_RELATIVE_PATH = 'data_test.csv'
SS_NN_ABSOLUTE_PATH = os.path.join(CWD, SS_NN_RELATIVE_PATH)
DATA_ABSOLUTE_PATH = os.path.join(CWD, DATA_RELATIVE_PATH)

class Test():
    def __init__(self, attack_simple: Attack, attack_nn: Attack, batchsize=40) -> None:
        self.attack_simple = attack_simple
        self.attack_nn = attack_nn
        

    def _prep_simple_test(self, batchsize=40):
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT).eval()
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        fmodel = fb.models.pytorch.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

        images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=batchsize)
        data = Data(images, labels)

        return fmodel, data


    def _prep_nn_test(self):
        if os.path.exists(SS_NN_ABSOLUTE_PATH):
            ss_nn_pipeline = mlflow.sklearn.load_model(SS_NN_ABSOLUTE_PATH)
        else:
            print(f"The directory '{SS_NN_ABSOLUTE_PATH}' does not exist.")
            return None,  None
        
        fmodel = None
        data = None

        if ss_nn_pipeline is not None:
            standard_scaler_from_nn_pipeline = ss_nn_pipeline.steps[0][1]
            nn_model = ss_nn_pipeline.steps[1][1].module_
            fmodel = fb.models.pytorch.PyTorchModel(nn_model, bounds=(-2., 30000.))

            if os.path.exists(DATA_ABSOLUTE_PATH):
                csv_filename = DATA_ABSOLUTE_PATH
            else:
                return None,  None

            data_df = pd.read_csv(csv_filename, skiprows=[0], nrows=10000)
            data_arr = data_df.values
            data = torch.tensor(data_arr, requires_grad=False, dtype=torch.float)
            data, result = torch.hsplit(data, [91, ])
            result = torch.tensor(result, requires_grad=False, dtype=torch.long)
            data = Data(data, result)


        return fmodel, data


    def _conduct(self, attack: FoolboxAttack, model, data: Data):
        time_start = time.time()

        if isinstance(model, PyTorchModel):
            print(f"Model accuracy before attack: {fb.utils.accuracy(model, data.input, data.output)}")
        print(f"Starting attack. ({time.asctime(time.localtime(time_start))})")
        adversarials = None

        adversarials = attack.conduct(model, data)
        # try:
        #     adversarials = attack.conduct(model, data)
        # except Exception as e:
        #     print(f"Error: {e}")

        time_end = time.time()
        print(f"Attack done. ({time.asctime(time.localtime(time_end))})")
        print(f"Took {time_end - time_start}\n")

        if adversarials is not None and isinstance(model, PyTorchModel):
            print(f"Model accuracy after attack: {accuracy(model, adversarials, data.output)}")

        return adversarials

    def test_simple(self):
        model, data = self._prep_simple_test()
        return self._conduct(self.attack_simple, model, data)
        

    def test_nn(self):
        model, data = self._prep_nn_test()
        if data is None or model is None:
            return None
        return self._conduct(self.attack_nn, model, data)
    