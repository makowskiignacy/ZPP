import getpass
import mlflow
import os

import torchvision.models as tv_models
import foolbox as fb
import art
from torch import nn, optim

from attacks.helpers.data import Data
from attacks.helpers.parameters import FoolboxParameters, ARTParameters
from utils.dataloader import DataLoader


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

    generic_parameters_bb = {"epsilon_rate": 0.01}
    attack_specific_parameters_bb = {"lr": 10, 'steps': 100}
    foolbox_parameters_bb = FoolboxParameters(attack_specific_parameters_bb, generic_parameters_bb)

    generic_parameters_bi = {"epsilon_rate": 0.05}
    attack_specific_parameters_bi = {"steps": 10, "random_start": True}
    foolbox_parameters_bi = FoolboxParameters(attack_specific_parameters_bi, generic_parameters_bi)

    generic_parameters_pgd = {"epsilon_rate": 0.01}
    attack_specific_parameters_pgd = {"steps": 100, "random_start": True}
    foolbox_parameters_pgd = FoolboxParameters(attack_specific_parameters_pgd, generic_parameters_pgd)

    generic_parameters_sap = {"epsilon_rate": 0.01}
    attack_specific_parameters_sap = {"steps": 10, "across_channels": True}
    foolbox_parameters_sap = FoolboxParameters(attack_specific_parameters_sap, generic_parameters_sap)

    foolbox_parameters = {"brendel_bethge": foolbox_parameters_bb,
                          "basic_iterative": foolbox_parameters_bi,
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
    CWD = os.getcwd()
    SS_NN_RELATIVE_PATH = 'ss_nn/'
    SS_NN_ABSOLUTE_PATH = os.path.join(CWD, SS_NN_RELATIVE_PATH)
    ss_nn_pipeline = mlflow.sklearn.load_model(SS_NN_ABSOLUTE_PATH)
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
            print(f"Pobieranie niepowiodło się z błędem:\n{e}\nMoże nie ma takiego pliku?")
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
