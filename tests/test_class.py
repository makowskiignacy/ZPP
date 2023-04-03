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

class Test:
    def __init__(self, attack_simple: Attack, attack_nn: Attack) -> None:
        self.attack_simple = attack_simple
        self.attack_nn = attack_nn

    def prep_simple_test(self, batchsize=40):
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT).eval()
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        fmodel = fb.models.pytorch.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

        images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=batchsize)
        data = Data(images, labels)

        self.fmodel = fmodel
        self.data = data


    def prep_nn_test(self):
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
                    if i < 10000:
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

            self.fmodel = fmodel
            self.data = data
        self.fmodel = None
        self.data = None


    def conduct(self):
        time_start = time.time()

        # print(type(model))
        if isinstance(self.fmodel, PyTorchModel):
            print(f"Model accuracy before attack: {fb.utils.accuracy(self.fmodel, self.data.input, self.data.output)}")
        print(f"Starting attack. ({time.asctime(time.localtime(time_start))})")

        adversarials = self.attack_simple.conduct(self.fmodel, self.data)

        time_end = time.time()
        print(f"Attack done. ({time.asctime(time.localtime(time_end))})")
        print(f"Took {time_end - time_start}\n")

        if adversarials is not None and isinstance(self.fmodel, PyTorchModel):
            print(f"Model accuracy after attack: {accuracy(self.fmodel, adversarials, self.data.output)}")

        return adversarials
