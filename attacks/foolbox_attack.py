from attacks.attack import Attack
from foolbox.models.pytorch import PyTorchModel
from foolbox.models.numpy import NumPyModel
from foolbox.models.tensorflow import TensorFlowModel
from foolbox.models.jax import JAXModel
import torch
import keras

class FoolboxAttack(Attack):

    def __init__(self, args):
        super().__init__()
        self.criterion = args.get("criterion")
        self.epsilon = args.get("epsilon", 0.01)
        self.min = args.get("min")
        self.max = args.get("max")

    @staticmethod
    def to_unified_format(data_from_attack):
        # Dla celów testowych lepiej żeby funkcja zwracała cokolwiek
        return data_from_attack

    def reformat_model(self, model):
        model2 = None
        bounds = (self.min, self.max)
        if isinstance(model, torch.nn.Module):
            model2 = PyTorchModel(model, bounds)
        elif isinstance(model, keras.Model):
            model2 = TensorFlowModel(model2, bounds)

        return model2

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
            print("ERROR")
            return 0

        return output


    def conduct(self, model, data):
        pass
