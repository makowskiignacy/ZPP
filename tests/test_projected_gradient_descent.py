import time
import mlflow
import csv
import numpy as np
import torch

import torchvision.models as tv_models
import foolbox as fb
from attacks.foolboxattacks.projected_gradient_descent import L1ProjectedGradientDescent, L2ProjectedGradientDescent, LinfProjectedGradientDescent
from attacks.foolboxattacks.projected_gradient_descent import L1AdamProjectedGradientDescent, L2AdamProjectedGradientDescent, LinfAdamProjectedGradientDescent
from attacks.helpers.data import Data
from attacks.helpers.parameters import FoolboxParameters
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


def main():
    attack_specific_parameters = {"steps": 100, "random_start": True}
    generic_parameters_simple = {"epsilon": 0.01}
    generic_parameters_nn = {"epsilon": 30}
    parameters_simple = FoolboxParameters(attack_specific_parameters,generic_parameters_simple)
    parameters_nn = FoolboxParameters(attack_specific_parameters, generic_parameters_nn)

    attack_pgd1 = L1ProjectedGradientDescent(parameters_simple)
    attack_pgd2 = L2ProjectedGradientDescent(parameters_simple)
    attack_pgdinf = LinfProjectedGradientDescent(parameters_simple)
    attack_pgd1_a = L1AdamProjectedGradientDescent(parameters_simple)
    attack_pgd2_a = L2AdamProjectedGradientDescent(parameters_simple)
    attack_pgdinf_a = LinfAdamProjectedGradientDescent(parameters_simple)

    smodel, sdata = simple_test(batchsize=20)

    print("Attack pgd1 simple")
    result1s = conduct(attack_pgd1, smodel, sdata)

    print("Attack pgd2 simple")
    result2s = conduct(attack_pgd2, smodel, sdata)

    print("Attack pgdinf simple")
    resultinfs = conduct(attack_pgdinf, smodel, sdata)

    print("Attack pgd1 with Adam simple")
    result1as = conduct(attack_pgd1_a, smodel, sdata)

    print("Attack pgd2 with Adam simple")
    result2as = conduct(attack_pgd2_a, smodel, sdata)

    print("Attack pgdinf with Adam simple")
    resultinfas = conduct(attack_pgdinf_a, smodel, sdata)

    nn_model, nn_data = nn_test()
    if nn_model is not None and nn_data is not None:
        attack_pgd1 = L1ProjectedGradientDescent(parameters_nn)
        attack_pgd2 = L2ProjectedGradientDescent(parameters_nn)
        attack_pgdinf = LinfProjectedGradientDescent(parameters_nn)
        attack_pgd1_a = L1AdamProjectedGradientDescent(parameters_nn)
        attack_pgd2_a = L2AdamProjectedGradientDescent(parameters_nn)
        attack_pgdinf_a = LinfAdamProjectedGradientDescent(parameters_nn)

        print("Attack pgd1 nn")
        result1nn = conduct(attack_pgd1, nn_model, nn_data)

        print("Attack pgd2 nn")
        result2nn = conduct(attack_pgd2, nn_model, nn_data)

        print("Attack pgdinf nn")
        resultinfnn = conduct(attack_pgdinf, nn_model, nn_data)

        print("Attack pgd1 with Adam nn")
        result1ann = conduct(attack_pgd1_a, nn_model, nn_data)

        print("Attack pgd2 with Adam nn")
        result2ann = conduct(attack_pgd2_a, nn_model, nn_data)

        print("Attack pgdinf with Adam nn")
        resultinfann = conduct(attack_pgdinf_a, nn_model, nn_data)


if __name__ == '__main__':
    main()
