from attacks.art_attack import ARTAttack

from art.estimators.classification import KerasClassifier, PyTorchClassifier, SklearnClassifier, TensorFlowV2Classifier
from art.attacks.evasion.adversarial_patch.adversarial_patch_numpy import AdversarialPatchNumpy
from art.attacks.evasion.adversarial_patch.adversarial_patch_tensorflow import AdversarialPatchTensorFlowV2
from art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
from typing import Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from utils.logger import test_logger

class AdversarialPatch(ARTAttack):
    def __init__(self, parameters):
        super().__init__(parameters.classifier_parameters)
        self.attack = None
        self._attack_params = {
            "classifier": parameters.attack_parameters.get("classifier"),
            # rotation_max (float) – The maximum rotation applied to random patches.
            # The value is expected to be in the range [0, 180].
            "rotation_max": parameters.attack_parameters.get("rotation_max", 22.5),
            # scale_min (float) – The minimum scaling applied to random patches.
            # The value should be in the range [0, 1], but less than scale_max.
            "scale_min": parameters.attack_parameters.get("scale_min", 0.1),
            "scale_max": parameters.attack_parameters.get("scale_max", 1.0),
            # learning_rate (float) – The learning rate of the optimization.
            "learning_rate": parameters.attack_parameters.get("learning_rate", 5.0),
            # max_iter (int) – The number of optimization steps.
            "max_iter": parameters.attack_parameters.get("max_iter", 500),
            # batch_size (int) – The size of the training batch.
            "batch_size": parameters.attack_parameters.get("batch_size", 16),
            # patch_shape – The shape of the adversarial patch as a tuple of shape (width, height, nb_channels).
            # Currently only supported for TensorFlowV2Classifier. For classifiers of other frameworks the patch_shape
            # is set to the shape of the input samples.
            "patch_shape": parameters.attack_parameters.get("patch_shape"),
            "targeted": parameters.attack_parameters.get("targeted", True),
            "verbose": parameters.attack_parameters.get("verbose", True)
        }

        self._attack: Union[AdversarialPatchTensorFlowV2, AdversarialPatchPyTorch, AdversarialPatchNumpy]
        if isinstance(self._attack_params.get('classifier'), TensorFlowV2Classifier):
            self._attack = AdversarialPatchTensorFlowV2(
                classifier=self._attack_params.get('classifier'),
                rotation_max=self._attack_params.get('rotation_max'),
                scale_min=self._attack_params.get('scale_min'),
                scale_max=self._attack_params.get('scale_max'),
                learning_rate=self._attack_params.get('learning_rate'),
                max_iter=self._attack_params.get('max_iter'),
                batch_size=self._attack_params.get('batch_size'),
                patch_shape=self._attack_params.get('patch_shape'),
                targeted=self._attack_params.get('targeted'),
                verbose=self._attack_params.get('verbose'),
            )
        elif isinstance(self._attack_params.get('classifier'), PyTorchClassifier):
            if self._attack_params.get('patch_shape') is not None:
                self._attack = AdversarialPatchPyTorch(
                    estimator=self._attack_params.get('classifier'),
                    rotation_max=self._attack_params.get('rotation_max'),
                    scale_min=self._attack_params.get('scale_min'),
                    scale_max=self._attack_params.get('scale_max'),
                    distortion_scale_max=0.0,
                    learning_rate=self._attack_params.get('learning_rate'),
                    max_iter=self._attack_params.get('max_iter'),
                    batch_size=self._attack_params.get('batch_size'),
                    patch_shape=self._attack_params.get('patch_shape'),
                    patch_type="circle",
                    targeted=self._attack_params.get('targeted'),
                    verbose=self._attack_params.get('verbose'),
                )
            else:
                raise ValueError("`patch_shape` cannot be `None` for `AdversarialPatchPyTorch`.")
        else:
            self._attack = AdversarialPatchNumpy(
                classifier=self._attack_params.get('classifier'),
                rotation_max=self._attack_params.get('rotation_max'),
                scale_min=self._attack_params.get('scale_min'),
                scale_max=self._attack_params.get('scale_max'),
                learning_rate=self._attack_params.get('learning_rate'),
                max_iter=self._attack_params.get('max_iter'),
                batch_size=self._attack_params.get('batch_size'),
                targeted=self._attack_params.get('targeted'),
                verbose=self._attack_params.get('verbose'),
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
        test_logger.debug("Creating adversarial patch.")

        if len(x.shape) == 2:  # pragma: no cover
            raise ValueError(
                "Feature vectors detected. The adversarial patch can only be applied to data with spatial "
                "dimensions."
            )

        return self._attack.generate(x=x, y=y, **kwargs)

    def conduct(self, model, data):
        self._set_classifier(model, data)
        self.attack = AdversarialPatch(**self._attack_params)
        return super().conduct(model, data)
