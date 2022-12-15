from attacks.artattacks.adversarial_patch import AdversarialPatch
from attacks.foolboxattacks.fast_gradient_sign import FastGradientSign

import mlflow
import csv
import numpy as np
import torch


class Data:
    def __init__(self, x, y):
        self.input = x
        self.output = y


ss_nn_pipeline = mlflow.sklearn.load_model('ss_nn/')
standard_scaler_from_nn_pipeline = ss_nn_pipeline.steps[0][1]
nn_model = ss_nn_pipeline.steps[1][1].module_

dict = {"epsilon": 0.01}
attack = FastGradientSign(dict)
csv_filename = 'data_small.csv'

with open(csv_filename) as f:
    reader = csv.reader(f)
    data = list(tuple(line) for line in reader)
    data.pop(0)
    data = np.array(data, dtype=int)
    data, result = np.hsplit(data, [91])
    print(data.shape)
    print(result.shape)
    data = Data(data, result)
    result = attack.conduct(nn_model, data)
    print(result)

