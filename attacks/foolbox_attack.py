from attacks.attack import Attack
from foolbox.models.pytorch import PyTorchModel
from foolbox.models.numpy import NumPyModel
from foolbox.models.tensorflow import TensorFlowModel
from foolbox.models.jax import JAXModel
from foolbox.utils import accuracy
import time
import foolbox as fb
import torch
import keras
from attacks.helpers.data import Data
from eagerpy.astensor import astensor
from utils.logger import test_logger


class FoolboxAttack(Attack):
    '''
    Klasa abstrakcyjna Ataku dla ataków zaimplementowanych w bibliotece Foolbox.
    '''

    def __init__(self, args):
        super().__init__()
        self.criterion_type = args.get("criterion_type", "misclassification")
        self.epsilon = args.get("epsilon")
        self.epsilon_rate = args.get("epsilon_rate")
        self.min = args.get("min")
        self.max = args.get("max")
        self.criterion = None

    def verify_bounds(self, data: Data):
        if self.min is not None and self.max is not None:
            return
        input_values = astensor(data.input)
        self.min = input_values.min().item()
        self.max = input_values.max().item()


    def verify_epsilon(self):
        if self.epsilon is not None:
            return
        elif self.epsilon_rate is not None:
            self.epsilon = (self.max - self.min)*self.epsilon_rate
        else:
            self.epsilon = (self.max-self.min)*0.001

    @staticmethod
    def to_unified_format(data_from_attack):
        '''
        Metoda unifikująca dane wyjściowe ataku z odgórnie ustalonym formatem
        danych.

        Parametry
        ---------
        data_from_attack
            Dane otrzymane po przeprowadzeniu ataku.
        
        Wyjście
        -------
            Dane otrzymane po przeprowadzeniu ataku w zunifikowanym formacie,
            gotowe do dalszej obróbki.
        '''
        # Dla celów testowych lepiej żeby funkcja zwracała cokolwiek
        return data_from_attack

    def reformat_model(self, model):
        reformatted_model = model
        bounds = (self.min, self.max)
        if isinstance(model, torch.nn.Module):
            reformatted_model = PyTorchModel(model, bounds)
        elif isinstance(model, keras.Model):
            reformatted_model = TensorFlowModel(reformatted_model, bounds)

        return reformatted_model

    # This makes sure that the output of experiments used for training has a 1D format,
    # instead of a faux-2D format, where one of the dimensions has a length of 1.
    def flatten_output(self,data):
        if len(data.output.shape) == 2 and len(data.output[1, :]) == 1:
            output = data.output[:, 0]
        elif len(data.output.shape) == 2 and len(data.output[:, 1]) == 1:
            output = data.output[0, :]
        elif len(data.output.shape) == 1:
            output = data.output
        else:
            test_logger.error("ERROR")
            return 0

        return output

    def conduct(self, model, data):
        pass

    def accuracy(self, model, input_data, output):
        if isinstance(model, PyTorchModel):
            return accuracy(model, input_data, output)
        else:
            return "No accuracy measure for this type of model"

