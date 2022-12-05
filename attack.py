from tensorflow import keras
import tensorflow as tf

import art
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import AdversarialPatch
from art.utils import load_mnist


class Attack:
    def __init__(self, args):
        pass

    def conduct(self, model, data):
        pass


class ARTAttack(Attack):
    @staticmethod
    def to_unified_format(data_from_attack):
        pass

    # x(ndarray)–An array with the original input images of shape NHWC or NCHW or input videos of shape NFHWC or NFCHW.
    # y – An array with the original true labels.
    # mask (np.ndarray) – A boolean array of shape equal to the shape of a single samples (1, H, W)
    # or the shape of x (N, H, W) without their channel dimensions.
    # Any features for which the mask is True can be the center location of the patch during sampling.
    # reset_patch (bool) – If True reset patch to initial values of mean of minimal and maximal clip value,
    # else if False (default) restart from previous patch values created by previous call to generate
    # or mean of minimal and maximal clip value if first call to generate.
    def __init__(self, args):
        super().__init__(args)

        self.classifier = None
        self.attack = None

        # Pytanie czy data i args powinny być osobnymi bytami,

        # tutaj jest pomieszanie z popląctaniem x=data pozostałe to parametry

        if "y" in args:
            self.y = args["rotation_max"]
        else:
            self.y = None

        if "mask" in args:
            self.mask = args["mask"]
        else:
            self.mask = None

        if "reset_patch" in args:
            self.reset_patch = args["reset_patch"]
        else:
            self.reset_patch = False

    def conduct(self, model, data):
        if self.y is not None and self.mask is not None:
            return ARTAttack.to_unified_format(self.attack.generate(x=data, y=self.y, mask=self.mask, reset_patch=self.reset_patch))
        return ARTAttack.to_unified_format(self.attack.generate(x=data, reset_patch=self.reset_patch))

    def set_classifier(self, model):
        if isinstance(model, keras.Model):
            self.classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value))
            self.classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
        else:
            raise Exception("Model type not supported")


class AdversarialPatch(ARTAttack):
    # rotation_max (float) – The maximum rotation applied to random patches.
    # The value is expected to be in the range [0, 180].

    # scale_min (float) – The minimum scaling applied to random patches.
    # The value should be in the range [0, 1], but less than scale_max.

    # scale_max (float) – The maximum scaling applied to random patches.
    # The value should be in the range [0, 1], but larger than scale_min.

    # learning_rate (float) – The learning rate of the optimization.

    # max_iter (int) – The number of optimization steps.

    # batch_size (int) – The size of the training batch.

    # patch_shape – The shape of the adversarial patch as a tuple of shape (width, height, nb_channels).
    # Currently only supported for TensorFlowV2Classifier. For classifiers of other frameworks the patch_shape
    # is set to the shape of the input samples.

    # targeted (bool) – Indicates whether the attack is targeted (True) or untargeted (False).

    # verbose (bool) – Show progress bars.

    def __init__(self, args):
        super().__init__(args)

        if "rotation_max" in args:
            self.rotation_max = args["rotation_max"]
        else:
            self.rotation_max = 22.5

        if "scale_min" in args:
            self.scale_min = args["scale_min"]
        else:
            self.scale_min = 0.1

        if "scale_max" in args:
            self.scale_max = args["scale_max"]
        else:
            self.scale_max = 1.0

        if "learning_rate" in args:
            self.learning_rate = args["learning_rate"]
        else:
            self.learning_rate = 5.0

        if "max_iter" in args:
            self.max_iter = args["max_iter"]
        else:
            self.max_iter = 500

        if "batch_size" in args:
            self.batch_size = args["batch_size"]
        else:
            self.batch_size = 16

        if "patch_shape" in args:
            self.patch_shape = args["patch_shape"]
        else:
            self.patch_shape = None

        if "targeted" in args:
            self.targeted = args["targeted"]
        else:
            self.targeted = True

        if "verbose" in args:
            self.verbose = args["verbose"]
        else:
            self.verbose = True

    def conduct(self, model, data):
        super().set_classifier(model)
        self.attack = AdversarialPatch(
            classifier=self.classifier,
            rotation_max=self.rotation_max,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            patch_shape=self.patch_shape,
            targeted=self.targeted,
            verbose=self.verbose)
        return super().conduct(model, data)