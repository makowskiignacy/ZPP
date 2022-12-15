from attacks.artattacks.adversarial_patch import AdversarialPatch

import mlflow
import csv
import numpy as np


class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y


ss_nn_pipeline = mlflow.sklearn.load_model('ss_nn/')
standard_scaler_from_nn_pipeline = ss_nn_pipeline.steps[0][1]
nn_model = ss_nn_pipeline.steps[1][1].module_
print(nn_model)
dict = {"classifier": nn_model.get_sklearn_object()}
attack = AdversarialPatch(dict)
csv_filename = 'data_test.csv'
with open(csv_filename) as f:
    reader = csv.reader(f)
    data = list(tuple(line) for line in reader)
    data = np.array(data)
    data, result = np.hsplit(data, [91])
    data = np.reshape(data, (353348, 13, 7))
    print(data.shape)
    print(result.shape)
    data = Data(data, result)
    attack.conduct(nn_model, data)


