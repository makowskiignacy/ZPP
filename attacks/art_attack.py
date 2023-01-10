from attacks.attack import Attack

import torch
import sklearn
from tensorflow import keras
from art.estimators.classification import KerasClassifier, PyTorchClassifier, SklearnClassifier
from mlplatformlib.model_building.binary.neural_network import BinaryNeuralNetworkClassifier


class ARTAttack(Attack):
    @staticmethod
    def to_unified_format(data_from_attack):
        return data_from_attack

    def __init__(self, args):
        super().__init__(args)

        self.classifier = None
        self.attack = None
        self.data = None

        # mask (np.ndarray) – A boolean array of shape equal to the shape of a single samples (1, H, W)
        # or the shape of x (N, H, W) without their channel dimensions.
        # Any features for which the mask is True can be the center location of the patch during sampling.
        if "mask" in args:
            self.mask = args["mask"]
        else:
            self.mask = None

        # reset_patch (bool) – If True reset patch to initial values of mean of minimal and maximal clip value,
        # else if False (default) restart from previous patch values created by previous call to generate
        # or mean of minimal and maximal clip value if first call to generate.
        if "reset_patch" in args:
            self.reset_patch = args["reset_patch"]
        else:
            self.reset_patch = False

        if "input_shape" in args:
            self.input_shape = args["input_shape"]
        else:
            self.input_shape = None

        if "loss" in args:
            self.loss = args["loss"]
        else:
            self.loss = None

        if "nb_classes" in args:
            self.nb_classes = args["nb_classes"]
        else:
            self.nb_classes = None

    def conduct(self, model, data):
        if self.data is not None and self.mask is not None:
            return ARTAttack.to_unified_format(self.attack.generate(
                x=self.data.input, y=self.data.output, mask=self.mask, reset_patch=self.reset_patch))
        return ARTAttack.to_unified_format(self.attack.generate(x=data.x, y=data.y, reset_patch=self.reset_patch))

    def set_data(self, data):
        self.data = data

    def set_classifier(self, model):
        if isinstance(model, keras.Model):
            # Setting classifier for Keras model, todo adding optional parameters
            self.classifier = KerasClassifier(model=model)
        elif isinstance(model, BinaryNeuralNetworkClassifier):
            self.classifier = model.get_sklearn_object()
        elif isinstance(model, torch.nn.Module):
            # Pytorch model
            if self.input_shape is None or self.loss is None or self.nb_classes is None:
                raise Exception("PyTorch model needs input_shape, loss and nb_classes to conduct attack")
            self.classifier = PyTorchClassifier(
                model=model, input_shape=self.input_shape, loss=self.loss, nb_classes=self.nb_classes)
        elif isinstance(model, sklearn.base.BaseEstimator):
            # todo adding optional parameters for sklearn model classifier
            self.classifier = SklearnClassifier(model=model)
        else:
            # TODO: Implementation of TensorFlow (and V2) classifiers, BlackBox
            raise Exception("Model type not supported")
