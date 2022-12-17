from attacks.attack import Attack
from foolbox.models.pytorch import PyTorchModel
from foolbox.models.numpy import NumPyModel
from foolbox.models.tensorflow import TensorFlowModel
from foolbox.models.jax import JAXModel
import torch


class FoolboxAttack(Attack):

    def __init__(self, args):
        super().__init__(args)
        if "criterion" in args:
            self.criterion = args["criterion"]
        else:
            self.criterion = None

        if "epsilon" in args:
            self.epsilon = args["epsilon"]
        else:
            self.epsilon = None

        if "min" in args:
            self.min = args["min"]
        else:
            self.min = -2

        if "max" in args:
            self.max = args["max"]
        else:
            self.max = 30000

    @staticmethod
    def to_unified_format(data_from_attack):
        pass

    def reformat_model(self, model):
        model2 = None
        bounds = (self.min, self.max)
        if isinstance(model, torch.nn.Module):
            model2 = PyTorchModel(model, bounds)

        return model2



    def conduct(self, model, data):
        pass
