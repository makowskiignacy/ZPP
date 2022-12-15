from attacks.art_attack import ARTAttack

from art.estimators.classification import KerasClassifier, PyTorchClassifier, SklearnClassifier, TensorFlowV2Classifier
from art.attacks.evasion.adversarial_patch.adversarial_patch_numpy import AdversarialPatchNumpy
from art.attacks.evasion.adversarial_patch.adversarial_patch_tensorflow import AdversarialPatchTensorFlowV2
from art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
from typing import Optional, Tuple, Union, TYPE_CHECKING
import numpy as np


class AdversarialPatch(ARTAttack):
    def __init__(self, args):
        super().__init__(args)

        if "classifier" in args:
            self.classifier = args["classifier"]

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

        self._attack: Union[AdversarialPatchTensorFlowV2, AdversarialPatchPyTorch, AdversarialPatchNumpy]
        if isinstance(self.classifier, TensorFlowV2Classifier):
            self._attack = AdversarialPatchTensorFlowV2(
                classifier=self.classifier,
                rotation_max=self.rotation_max,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                patch_shape=self.patch_shape,
                targeted=self.targeted,
                verbose=self.verbose,
            )
        elif isinstance(self.classifier, PyTorchClassifier):
            if self.patch_shape is not None:
                self._attack = AdversarialPatchPyTorch(
                    estimator=self.classifier,
                    rotation_max=self.rotation_max,
                    scale_min=self.scale_min,
                    scale_max=self.scale_max,
                    distortion_scale_max=0.0,
                    learning_rate=self.learning_rate,
                    max_iter=self.max_iter,
                    batch_size=self.batch_size,
                    patch_shape=self.patch_shape,
                    patch_type="circle",
                    targeted=self.targeted,
                    verbose=self.verbose,
                )
            else:
                raise ValueError("`patch_shape` cannot be `None` for `AdversarialPatchPyTorch`.")
        else:
            self._attack = AdversarialPatchNumpy(
                classifier=self.classifier,
                rotation_max=self.rotation_max,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                targeted=self.targeted,
                verbose=self.verbose,
            )

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
