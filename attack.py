import torch
import sklearn
import mlflow
import csv

from tensorflow import keras
from art.estimators.classification import KerasClassifier, PyTorchClassifier, SklearnClassifier, TensorFlowV2Classifier

from mlplatformlib.model_building.binary.neural_network import BinaryNeuralNetworkClassifier
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np


class Attack:
    def __init__(self, args):
        pass

    def conduct(self, model, data):
        pass


class ARTAttack(Attack):
    @staticmethod
    def to_unified_format(data_from_attack):
        pass

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
                x=self.data.x, y=self.data.y, mask=self.mask, reset_patch=self.reset_patch))
        return ARTAttack.to_unified_format(self.attack.generate(x=data, reset_patch=self.reset_patch))

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


class AdversarialPatch(ARTAttack):
    def __init__(self, args):
        super().__init__(args)

        # rotation_max (float) – The maximum rotation applied to random patches.
        # The value is expected to be in the range [0, 180].
        if "rotation_max" in args:
            self.rotation_max = args["rotation_max"]
        else:
            self.rotation_max = 22.5

        # scale_min (float) – The minimum scaling applied to random patches.
        # The value should be in the range [0, 1], but less than scale_max.
        if "scale_min" in args:
            self.scale_min = args["scale_min"]
        else:
            self.scale_min = 0.1

        if "scale_max" in args:
            self.scale_max = args["scale_max"]
        else:
            self.scale_max = 1.0

        # learning_rate (float) – The learning rate of the optimization.
        if "learning_rate" in args:
            self.learning_rate = args["learning_rate"]
        else:
            self.learning_rate = 5.0

        # max_iter (int) – The number of optimization steps.
        if "max_iter" in args:
            self.max_iter = args["max_iter"]
        else:
            self.max_iter = 500

        # batch_size (int) – The size of the training batch.
        if "batch_size" in args:
            self.batch_size = args["batch_size"]
        else:
            self.batch_size = 16

        # patch_shape – The shape of the adversarial patch as a tuple of shape (width, height, nb_channels).
        # Currently only supported for TensorFlowV2Classifier. For classifiers of other frameworks the patch_shape
        # is set to the shape of the input samples.
        if "patch_shape" in args:
            self.patch_shape = args["patch_shape"]
        else:
            self.patch_shape = None

        # targeted (bool) – Indicates whether the attack is targeted (True) or untargeted (False).
        if "targeted" in args:
            self.targeted = args["targeted"]
        else:
            self.targeted = True

        # verbose (bool) – Show progress bars.
        if "verbose" in args:
            self.verbose = args["verbose"]
        else:
            self.verbose = True

    def generate( self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate an adversarial patch and return the patch and its mask in arrays.

        :param x: An array with the original input images of shape NHWC or NCHW or input videos of shape NFHWC or NFCHW.
        :param y: An array with the original true labels.
        :param mask: A boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :type mask: `np.ndarray`
        :param reset_patch: If `True` reset patch to initial values of mean of minimal and maximal clip value, else if
                            `False` (default) restart from previous patch values created by previous call to `generate`
                            or mean of minimal and maximal clip value if first call to `generate`.
        :type reset_patch: bool
        :return: An array with adversarial patch and an array of the patch mask.
        """
        print("Creating adversarial patch.")

        if len(x.shape) == 2:  # pragma: no cover
            raise ValueError(
                "Feature vectors detected. The adversarial patch can only be applied to data with spatial "
                "dimensions."
            )

        return self._attack.generate(x=x, y=y, **kwargs)

    def conduct(self, model, data):
        super().set_classifier(model)
        args = {
            "classifier": self.classifier,
            "rotation_max": self.rotation_max,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "patch_shape": self.patch_shape,
            "targeted": self.targeted,
            "verbose": self.verbose
        }
        self.attack = AdversarialPatch(args)
        return super().conduct(model, data)


ss_nn_pipeline = mlflow.sklearn.load_model('ss_nn/')
standard_scaler_from_nn_pipeline = ss_nn_pipeline.steps[0][1]
nn_model = ss_nn_pipeline.steps[1][1].module_
print(nn_model)
dict = ""
attack = AdversarialPatch(dict)
csv_filename = 'data_test.csv'
with open(csv_filename) as f:
    reader = csv.reader(f)
    data = list(tuple(line) for line in reader)
    data = np.array(data)
    print(data.shape)
    attack.conduct(nn_model, data)


