import time
import mlflow
import csv
import numpy as np
import torch

import torchvision.models as tv_models
import foolbox as fb
from attacks.foolboxattacks.brendel_bethge import L0BrendelBethge, L1BrendelBethge, L2BrendelBethge, LinfinityBrendelBethge
from attacks.helpers.data import Data
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
    ss_nn_pipeline = mlflow.sklearn.load_model('/ss_nn/')
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


        
def main():
    bb_simple_args_dict = {'epsilons': 1, "lr": 1, 'steps': 100, 'min': 0, 'max': 1}
    bb_nn_args_dict = {'epsilons': 1, "lr": 1, 'steps': 100, 'min': -1.0, 'max': 28157.0}

    attack_bb0_simple = L0BrendelBethge(bb_simple_args_dict)
    attack_bb1_simple = L1BrendelBethge(bb_simple_args_dict)
    attack_bb2_simple = L2BrendelBethge(bb_simple_args_dict)
    attack_bbinf_simple = LinfinityBrendelBethge(bb_simple_args_dict)

    attack_bb1_nn = L1BrendelBethge(bb_nn_args_dict)
    attack_bb2_nn = L2BrendelBethge(bb_nn_args_dict)
    attack_bbinf_nn = LinfinityBrendelBethge(bb_nn_args_dict)

    smodel, sdata = simple_test()

    print("Attack bb0 simple")
    result0 = conduct(attack_bb0_simple, smodel, sdata)
    print(result0)

    # print("Attack bb1 simple")
    # result1 = conduct(attack_bb1_simple, smodel, sdata)
    # print(result1)

    # print("Attack bb2 simple")
    # result2s = conduct(attack_bb2_simple, smodel, sdata)
    # print(result2s)

    # print("Attack bbinf simple")
    # resultinf = conduct(attack_bbinf_simple, smodel, sdata)
    # print(resultinf)

    nn_model, nn_data = nn_test()
    if nn_model is not None and nn_data is not None:
        print("Attack bb1 nn")
        result1nn = conduct(attack_bb1_nn, nn_model, nn_data)
        print(result1nn)
        
        print("Attack bb2 nn")
        result2nn = conduct(attack_bb2_nn, nn_model, nn_data)
        print(result2nn)
        
        print("Attack bbinf nn")
        resultinfnn = conduct(attack_bbinf_nn, nn_model, nn_data)
        print(resultinfnn)

if __name__ == '__main__':
    main()
