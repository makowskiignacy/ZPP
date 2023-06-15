import copy
import csv
import getpass
import copy

from tensorflow import keras
import mlflow
import os
import numpy
import pandas as pd
import torch
import timm
import detectors
import pickle

import eagerpy as ep
import torchvision.models as tv_models
import foolbox as fb
import art
import art.estimators.classification.pytorch # TODO remove 
from torch import Tensor, nn, optim
from foolbox.models.base import Model
from typing import Callable
from foolbox.attacks.gradient_descent_base import BaseGradientDescent

from attacks.helpers.data import Data
from attacks.helpers.parameters import FoolboxParameters, ARTParameters
from utils.logger import test_logger
from utils.config import SS_INPUT_ROWS
from utils.dataloader import DataLoader


def resnet18_cifar100_input(batchsize = 100):
    model = timm.create_model("resnet18_cifar100", pretrained=True)
    foolbox_model = fb.models.pytorch.PyTorchModel(model=model, bounds=(-1, 256))
    art_model = art.estimators.classification.pytorch.PyTorchClassifier(model=model, clip_values=(-1, 256), input_shape=(3, 32, 32), loss=nn.CrossEntropyLoss(), nb_classes=100)
    art_criterion = nn.CrossEntropyLoss()
    art_optimizer = optim.Adam(art_model._model.parameters(), lr=0.01)

    cifar100_data_path = 'tests/test_files/cifar100_testing_data'
    with open(cifar100_data_path, 'rb') as file:
        whole_data = pickle.load(file, encoding='latin1')
    input_data = whole_data['data']
    input_data = input_data.reshape(len(input_data), 3, 32, 32)
    input_data = input_data[:batchsize, :, :, :]
    output_data = numpy.array(whole_data['fine_labels'])
    output_data = output_data[:batchsize]
    foolbox_data = Data(torch.from_numpy(input_data).float(), torch.from_numpy(output_data).long())
    art_data = Data(input_data, output_data)
    classifier_parameters_default = {"clip_values": (-1, 256), "loss": art_criterion, "optimizer": art_optimizer, "input_shape": (3, 32, 32), "nb_classes": 100}

    predictions = art_model.predict(input_data, training_mode=True)
    strongest_prediction = numpy.argmax(predictions, axis=1)
    correct = numpy.sum(strongest_prediction == output_data)
    accuracy = correct / len(output_data)
    print(accuracy)

    attack_parameters_default = {"verbose": True}
    art_parameters_default = ARTParameters(classifier_parameters_default, attack_parameters_default)
    attack_parameters_joker = {}
    art_parameters_joker = ARTParameters(classifier_parameters_default, attack_parameters_joker)

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

    generic_parameters_default = {"epsilon_rate": 0.01}

    attack_specific_parameters_an = {}
    foolbox_parameters_an = FoolboxParameters(attack_specific_parameters_an, generic_parameters_default)

    attack_specific_parameters_bb = {"lr": 10, 'steps': 100}
    foolbox_parameters_bb = FoolboxParameters(attack_specific_parameters_bb, generic_parameters_default)

    generic_parameters_bi = {"epsilon_rate": 0.05}
    attack_specific_parameters_bi = {"steps": 10, "random_start": True}
    foolbox_parameters_bi = FoolboxParameters(attack_specific_parameters_bi, generic_parameters_bi)

    attack_specific_parameters_cw = {"steps": 100}
    foolbox_parameters_cw = FoolboxParameters(attack_specific_parameters_cw, generic_parameters_default)

    attack_specific_parameters_nf = {"steps": 100, "stepsize": 100}
    foolbox_parameters_nf = FoolboxParameters(attack_specific_parameters_nf, generic_parameters_default)

    attack_specific_parameters_pgd = {"steps": 100, "random_start": True}
    foolbox_parameters_pgd = FoolboxParameters(attack_specific_parameters_pgd, generic_parameters_default)

    attack_specific_parameters_sap = {"steps": 10, "across_channels": True}
    foolbox_parameters_sap = FoolboxParameters(attack_specific_parameters_sap, generic_parameters_default)

    foolbox_parameters = {"additive_noise": foolbox_parameters_an,
                          "brendel_bethge": foolbox_parameters_bb,
                          "basic_iterative": foolbox_parameters_bi,
                          "carlini_wagner": foolbox_parameters_cw,
                          "newton_fool": foolbox_parameters_nf,
                          "projected_gradient_descent": foolbox_parameters_pgd,
                          "salt_and_pepper": foolbox_parameters_sap}

    return foolbox_model, model, foolbox_data, art_data, foolbox_parameters, art_parameters


def simple_input(batchsize=4):
    model = tv_models.resnet18(pretrained=True)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    foolbox_model = fb.models.pytorch.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    images, labels = fb.utils.samples(foolbox_model, dataset='imagenet', batchsize=batchsize)
    foolbox_data = Data(images, labels)
    test_logger.debug(f"############################ simple res labels: {numpy.shape(labels)} {labels}")
    art_data = Data(images.numpy(), labels.numpy())
    art_model = art.estimators.classification.pytorch.PyTorchClassifier(model=model, input_shape=(3, 224, 224),
                                                                        loss=nn.CrossEntropyLoss(),
                                                                        nb_classes=(art_data.output.max() - min(0,
                                                                                                                art_data.output.min()) + 1))

    predictions = art_model.predict(images.numpy(), training_mode=True)
    strongest_prediction = numpy.argmax(predictions, axis=1)
    correct = numpy.sum(strongest_prediction == labels.numpy())
    accuracy = correct / len(labels.numpy())
    print(accuracy)

    art_criterion = nn.CrossEntropyLoss()
    art_optimizer = optim.Adam(art_model._model.parameters(), lr=0.01)

    classifier_parameters_default = {"clip_values": (0, 1), "loss": art_criterion, "optimizer": art_optimizer,
                             "input_shape": (3, 224, 224),
                             "nb_classes": (art_data.output.max() - min(0, art_data.output.min()) + 1)}
    attack_parameters_default = {"verbose": True}
    art_parameters_default = ARTParameters(classifier_parameters_default, attack_parameters_default)

    attack_parameters_joker = {}
    art_parameters_joker = ARTParameters(classifier_parameters_default, attack_parameters_joker)

    generic_parameters_an = {"epsilon": 0.01}
    attack_specific_parameters_an = {}
    foolbox_parameters_an = FoolboxParameters(attack_specific_parameters_an, generic_parameters_an)

    generic_parameters_bb = {"epsilon_rate": 0.01}
    attack_specific_parameters_bb = {"lr": 10, 'steps': 100}
    foolbox_parameters_bb = FoolboxParameters(attack_specific_parameters_bb, generic_parameters_bb)

    generic_parameters_bi = {"epsilon_rate": 0.05}
    attack_specific_parameters_bi = {"steps": 10, "random_start": True}
    foolbox_parameters_bi = FoolboxParameters(attack_specific_parameters_bi, generic_parameters_bi)

    generic_parameters_cw = {"epsilon_rate": 0.01}
    attack_specific_parameters_cw = {"steps": 100}
    foolbox_parameters_cw = FoolboxParameters(attack_specific_parameters_cw, generic_parameters_cw)

    generic_parameters_nf = {"epsilon_rate": 0.01}
    attack_specific_parameters_nf = {"steps": 100, "stepsize": 100}
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
                          "newton_fool": foolbox_parameters_nf,
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


def is_binary_model(targets: Tensor):
    assert len(targets.shape) == 2
    return targets.shape[1] == 1


def custom_get_loss_fn(
        self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users

        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)

            m = torch.nn.Sigmoid()
            loss = torch.nn.BCELoss()
            target = torch.reshape(labels.raw, (labels.shape[0], 1))

            logits = model(inputs.raw)

            output = loss(m(logits), target)
            return ep.astensor(output)

        return loss_fn


def nn_input():
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "ss_nn/")
    data_path = os.path.join(cwd, 'data_test.csv')

    if os.path.exists(model_path):
        ss_nn_pipeline = mlflow.sklearn.load_model(model_path)
    else:
        test_logger.error("Can't find model directory.")
        exit(1)

    if os.path.exists(data_path):
        csv_filename = data_path
    else:
        test_logger.error("Can't find data file.")
        exit(1)

    if ss_nn_pipeline is None:
        test_logger.error("Can't load model.")
        exit(1)

    nn_model = ss_nn_pipeline.steps[1][1].module_
    fmodel = fb.models.pytorch.PyTorchModel(nn_model, bounds=(-2., 30000.))

    data_df = pd.read_csv(csv_filename, skiprows=[0], nrows=SS_INPUT_ROWS)
    data = data_df.values
    data = torch.tensor(data, requires_grad=False, dtype=torch.float)
    data, result = torch.hsplit(data, [91, ])
    result = torch.tensor(result, requires_grad=False, dtype=torch.float)
    foolbox_data = Data(data, result)


    if  is_binary_model(result):
        BaseGradientDescent.get_loss_fn = custom_get_loss_fn

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

    generic_parameters_an = {"epsilon": 0.01}
    attack_specific_parameters_an = {}
    foolbox_parameters_an = FoolboxParameters(attack_specific_parameters_an, generic_parameters_an)
    
    generic_parameters_bb = {"epsilon_rate": 0.01}
    attack_specific_parameters_bb = {"lr": 10, 'steps': 100}
    foolbox_parameters_bb = FoolboxParameters(attack_specific_parameters_bb, generic_parameters_bb)

    generic_parameters_bi = {"epsilon_rate": 0.05}
    attack_specific_parameters_bi = {"steps": 100, "random_start": True}
    foolbox_parameters_bi = FoolboxParameters(attack_specific_parameters_bi, generic_parameters_bi)

    generic_parameters_cw = {"epsilon_rate": 0.01}
    attack_specific_parameters_cw = {"steps": 100}
    foolbox_parameters_cw = FoolboxParameters(attack_specific_parameters_cw, generic_parameters_cw)

    generic_parameters_nf = {"epsilon_rate": 0.01}
    attack_specific_parameters_nf = {"steps": 100, "stepsize": 100}
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
                            "newton_fool": foolbox_parameters_nf,
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


def nn_input_cloud():
    ss_nn_pipeline = mlflow.sklearn.load_model("../ss_nn/")
    if ss_nn_pipeline is not None:
        nn_model = ss_nn_pipeline.steps[1][1].module_
        fmodel = fb.models.pytorch.PyTorchModel(nn_model, bounds=(-2., 30000.))

        nc_user = input("NextCloud username:")
        nc_pass = getpass.getpass("NextCloud password:")
        dataloader = DataLoader(
                    ml_user='test', ml_pass='test',
                    nc_user=nc_user, nc_pass=nc_pass,
                    ml_platform_address = 'ml-platform.test.rd.nask.pl',
                    ml_keycloak_address = 'auth.rd.nask.pl',
                    ml_realm  = 'ML-PLATFORM',
                    ml_client_id  = 'ml_platform',
                    nc_host  = 'https://nextcloud.cbitt.nask.pl/'
                )
        try:
            dataloader.download(dataloader.make_remote_path('data_test.csv'))
        except Exception as e:
            print(f"Pobieranie nie powiodło się z błędem:\n{e}\nMoże nie ma takiego pliku?")
            quit(1)
        data_list = dataloader.load_to_variable(
                libs=[DataLoader.SupportedLibrary.ART, DataLoader.SupportedLibrary.Foolbox],
                local_file_path=dataloader.make_local_path('data_test.csv'),
                number_of_samples=100)
        art_data = data_list[0]
        foolbox_data = data_list[1]
        art_model = art.estimators.classification.pytorch.PyTorchClassifier(model=nn_model,
                                                                            input_shape=art_data.input.shape,
                                                                            loss=nn.CrossEntropyLoss(),
                                                                            nb_classes=int(art_data.output.max() - min(0, art_data.output.min()) + 1))
        art_criterion = nn.CrossEntropyLoss()
        art_optimizer = optim.Adam(art_model._model.parameters(), lr=0.01)

        classifier_parameters_default = {"clip_values": (-2., 30000.), "loss": art_criterion,
                                         "optimizer": art_optimizer, "input_shape": art_data.input.shape,
                                         "nb_classes": int(art_data.output.max() - min(0, art_data.output.min()) + 1)}
        attack_parameters_default = {}
        art_parameters_default = ARTParameters(classifier_parameters_default, attack_parameters_default)

        generic_parameters_bb = {"epsilon_rate": 0.01}
        attack_specific_parameters_bb = {"lr": 10, 'steps': 100}
        foolbox_parameters_bb = FoolboxParameters(attack_specific_parameters_bb, generic_parameters_bb)

        generic_parameters_bi = {"epsilon_rate": 0.05}
        attack_specific_parameters_bi = {"steps": 100, "random_start": True}
        foolbox_parameters_bi = FoolboxParameters(attack_specific_parameters_bi, generic_parameters_bi)

        generic_parameters_pgd = {"epsilon_rate": 0.01}
        attack_specific_parameters_pgd = {"steps": 100, "random_start": True}
        foolbox_parameters_pgd = FoolboxParameters(attack_specific_parameters_pgd, generic_parameters_pgd)

        generic_parameters_sap = {"epsilon_rate": 0.01}
        attack_specific_parameters_sap = {"steps": 100, "across_channels": True}
        foolbox_parameters_sap = FoolboxParameters(attack_specific_parameters_sap, generic_parameters_sap)

        foolbox_parameters = {"brendel_bethge": foolbox_parameters_bb,
                              "basic_iterative": foolbox_parameters_bi,
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