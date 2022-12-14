from attacks.attack_v2 import Attack
from foolbox.models.pytorch import PyTorchModel
from foolbox.models.numpy import NumPyModel
from foolbox.models.tensorflow import TensorFlowModel
from foolbox.models.jax import JAXModel
import torch


class FoolboxAttack(Attack):

    def __init__(self, args):
        super().__init__()
        self.criterion = args.get("criterion")
        self.epsilon = args.get("epsilon", 0.0)
        self.min = args.get("min", -2)
        self.max = args.get("max", 30000)

    @staticmethod
    def to_unified_format(data_from_attack):
        # Dla celów testowych lepiej żeby funkcja zwracała cokolwiek
        return data_from_attack

    def reformat_model(self, model):
        model2 = None
        bounds = (self.min, self.max)
        if isinstance(model, torch.nn.Module):
            model2 = PyTorchModel(model, bounds)

        return model2

    def conduct(self, model, data):
        pass
