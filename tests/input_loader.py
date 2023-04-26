import csv
import mlflow
import numpy
import torch

import torchvision.models as tv_models
import foolbox as fb
import art
from torch import nn, optim

from attacks.helpers.data import Data
from attacks.helpers.parameters import FoolboxParameters, ARTParameters


def simple_input(batchsize=4):
    model = tv_models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    foolbox_model = fb.models.pytorch.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    images, labels = fb.utils.samples(foolbox_model, dataset='imagenet', batchsize=batchsize)
    foolbox_data = Data(images, labels)
    art_data = Data(images.numpy(), labels.numpy())
    art_model = art.estimators.classification.pytorch.PyTorchClassifier(model=model, input_shape=(3, 224, 224),
                                                                        loss=nn.CrossEntropyLoss(),
                                                                        nb_classes=(art_data.output.max() - min(0,
                                                                                                                art_data.output.min()) + 1))
    art_criterion = nn.CrossEntropyLoss()
    art_optimizer = optim.Adam(art_model._model.parameters(), lr=0.01)

    classifier_parameters_default = {"clip_values": (0, 1), "loss": art_criterion, "optimizer": art_optimizer,
                             "input_shape": (3, 224, 224),
                             "nb_classes": (art_data.output.max() - min(0, art_data.output.min()) + 1)}
    attack_parameters_default = {"verbose": True}
    art_parameters_default = ARTParameters(classifier_parameters_default, attack_parameters_default)

    attack_parameters_joker = {}
    art_parameters_joker = ARTParameters(classifier_parameters_default, attack_parameters_joker)

    generic_parameters_an = {"epsilon_rate": 0.01}
    attack_specific_parameters_an = {"lr": 10, 'steps': 100}
    foolbox_parameters_an = FoolboxParameters(attack_specific_parameters_an, generic_parameters_an)

    generic_parameters_bb = {"epsilon_rate": 0.01}
    attack_specific_parameters_bb = {"lr": 10, 'steps': 100}
    foolbox_parameters_bb = FoolboxParameters(attack_specific_parameters_bb, generic_parameters_bb)

    generic_parameters_bi = {"epsilon_rate": 0.05}
    attack_specific_parameters_bi = {"steps": 10, "random_start": True}
    foolbox_parameters_bi = FoolboxParameters(attack_specific_parameters_bi, generic_parameters_bi)

    generic_parameters_cw = {"epsilon_rate": 0.01}
    attack_specific_parameters_cw = {"steps": 100, "random_start": True}
    foolbox_parameters_cw = FoolboxParameters(attack_specific_parameters_cw, generic_parameters_cw)

    generic_parameters_nf = {"epsilon_rate": 0.01}
    attack_specific_parameters_nf = {"steps": 100, "random_start": True}
    foolbox_parameters_nf = FoolboxParameters(attack_specific_parameters_nf, generic_parameters_nf)

    generic_parameters_pgd = {"epsilon_rate": 0.01}
    attack_specific_parameters_pgd = {"steps": 100, "random_start": True}
    foolbox_parameters_pgd = FoolboxParameters(attack_specific_parameters_pgd, generic_parameters_pgd)

    generic_parameters_sap = {"epsilon_rate": 0.01}
    attack_specific_parameters_sap = {"steps": 10, "across_channels": True}
    foolbox_parameters_sap = FoolboxParameters(attack_specific_parameters_sap, generic_parameters_sap)

    foolbox_parameters = {"additive_noise": foolbox_parameters_an,
                          "brendel_bethge": foolbox_parameters_bb,
                          "basic_iterative": foolbox_parameters_bi,
                          "carlini_wagner": foolbox_parameters_cw,
                          "newton_Fool": foolbox_parameters_nf,
                          "projected_gradient_descent": foolbox_parameters_pgd,
                          "salt_and_pepper": foolbox_parameters_sap}

    art_parameters = {"deep_fool": art_parameters_default,
                      "fast_gradient": art_parameters_default,
                      "geometric_decision_based": art_parameters_default,
                      "jacobian_saliency_map": art_parameters_default,
                      "joker": art_parameters_joker,
                      "shadow": art_parameters_default,
                      "sign_opt": art_parameters_default,
                      "square": art_parameters_default,
                      "threshold": art_parameters_default,
                      "zeroth_order_optimization": art_parameters_default}

    return foolbox_model, model, foolbox_data, art_data, foolbox_parameters, art_parameters


def nn_input():
    ss_nn_pipeline = mlflow.sklearn.load_model('../ss_nn/')
    if ss_nn_pipeline is not None:
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
                    i = i + 1
                    row2 = []
                    for place in range(len(row)):
                        row2.append(float(row[place]))
                    data2.append(row2)
            data = data2
            data = torch.tensor(data, requires_grad=False, dtype=torch.float)
            data, result = torch.hsplit(data, [91, ])
            result = torch.tensor(result, requires_grad=False, dtype=torch.float)
            foolbox_data = Data(data, result)
            art_data = Data(data.numpy(), result.numpy())

        art_model = art.estimators.classification.pytorch.PyTorchClassifier(model=nn_model,
                                                                            input_shape=art_data.input.shape,
                                                                            loss=nn.CrossEntropyLoss(), nb_classes=int(
                art_data.output.max() - min(0, art_data.output.min()) + 1))
        art_criterion = nn.CrossEntropyLoss()
        art_optimizer = optim.Adam(art_model._model.parameters(), lr=0.01)

        classifier_parameters_default = {"clip_values": (-2., 30000.), "loss": art_criterion,
                                         "optimizer": art_optimizer, "input_shape": art_data.input.shape,
                                         "nb_classes": int(art_data.output.max() - min(0, art_data.output.min()) + 1)}
        attack_parameters_default = {}
        art_parameters_default = ARTParameters(classifier_parameters_default, attack_parameters_default)

        generic_parameters_an = {"epsilon_rate": 0.01}
        attack_specific_parameters_an = {"lr": 10, 'steps': 100}
        foolbox_parameters_an = FoolboxParameters(attack_specific_parameters_an, generic_parameters_an)
        
        generic_parameters_bb = {"epsilon_rate": 0.01}
        attack_specific_parameters_bb = {"lr": 10, 'steps': 100}
        foolbox_parameters_bb = FoolboxParameters(attack_specific_parameters_bb, generic_parameters_bb)

        generic_parameters_bi = {"epsilon_rate": 0.05}
        attack_specific_parameters_bi = {"steps": 100, "random_start": True}
        foolbox_parameters_bi = FoolboxParameters(attack_specific_parameters_bi, generic_parameters_bi)

        generic_parameters_cw = {"epsilon_rate": 0.01}
        attack_specific_parameters_cw = {"steps": 100, "random_start": True}
        foolbox_parameters_cw = FoolboxParameters(attack_specific_parameters_cw, generic_parameters_cw)

        generic_parameters_nf = {"epsilon_rate": 0.01}
        attack_specific_parameters_nf = {"steps": 100, "random_start": True}
        foolbox_parameters_nf = FoolboxParameters(attack_specific_parameters_nf, generic_parameters_nf)

        generic_parameters_pgd = {"epsilon_rate": 0.01}
        attack_specific_parameters_pgd = {"steps": 100, "random_start": True}
        foolbox_parameters_pgd = FoolboxParameters(attack_specific_parameters_pgd, generic_parameters_pgd)

        generic_parameters_sap = {"epsilon_rate": 0.01}
        attack_specific_parameters_sap = {"steps": 100, "across_channels": True}
        foolbox_parameters_sap = FoolboxParameters(attack_specific_parameters_sap, generic_parameters_sap)

        foolbox_parameters = {"additive_noise": foolbox_parameters_an,
                              "brendel_bethge": foolbox_parameters_bb,
                              "basic_iterative": foolbox_parameters_bi,
                              "carlini_wagner": foolbox_parameters_cw,
                              "newton_Fool": foolbox_parameters_nf,
                              "projected_gradient_descent": foolbox_parameters_pgd,
                              "salt_and_pepper": foolbox_parameters_sap}

        art_parameters = {"deep_fool": art_parameters_default,
                          "fast_gradient": art_parameters_default,
                          "geometric_decision_based": art_parameters_default,
                          "jacobian_saliency_map": art_parameters_default,
                          "joker": art_parameters_default,
                          "shadow": art_parameters_default,
                          "sign_opt": art_parameters_default,
                          "square": art_parameters_default,
                          "threshold": art_parameters_default,
                          "zeroth_order_optimization": art_parameters_default}

        return fmodel, nn_model, foolbox_data, art_data, foolbox_parameters, art_parameters
    return None, None
