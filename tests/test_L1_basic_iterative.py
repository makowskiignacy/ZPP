from attacks.foolboxattacks.L1_basic_iterative import L1BasicIterative

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

dict = {"epsilon": 0.01, "random_start": True}
attack = L1BasicIterative(dict)

csv_filename = 'data_test.csv'

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

    # FIXME: RuntimeError: expected scalar type Long but found Float
    result = torch.tensor(result, requires_grad=False, dtype=torch.float)
    data = Data(data, result)

    print("Attack")
    result = attack.conduct(nn_model, data)
    print(result)
    print("Done!")
