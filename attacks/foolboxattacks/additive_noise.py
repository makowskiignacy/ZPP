from abc import ABC
from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.additive_noise import L2AdditiveGaussianNoiseAttack, L2AdditiveUniformNoiseAttack, L2ClippingAwareAdditiveGaussianNoiseAttack, L2ClippingAwareAdditiveUniformNoiseAttack, LinfAdditiveUniformNoiseAttack, L2RepeatedAdditiveGaussianNoiseAttack, L2RepeatedAdditiveUniformNoiseAttack, L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack, L2ClippingAwareRepeatedAdditiveUniformNoiseAttack, LinfRepeatedAdditiveUniformNoiseAttack



from attacks.helpers.data import Data
from eagerpy.astensor import astensor


class GenericAdditiveNoise(FoolboxAttack, ABC):
    """
    :param parent - attack specific class from foolbox.attacks.additive_noise module
    """
    def __init__(self, parent, parameters):
        self.parent = parent
        self.parent.__init__(self, **parameters.attack_specific_parameters)
        FoolboxAttack.__init__(self, parameters.generic_parameters)


    def verify_bounds(self, data: Data):
        if hasattr(self, 'min') and hasattr(self, 'max'):
            return
        
        originals, _ = astensor(data.input)
        self.min = originals.min().item()
        self.max = originals.max().item()

        return


    def conduct(self, model, data):
        self.verify_bounds(data=data)
        output = super().flatten_output(data)
        model_correct_format = super().reformat_model(model)

        if self.criterion_type == "targeted_misclassification":
            self.criterion = TargetedMisclassification(output)
        if self.criterion_type == "misclassification":
            self.criterion = Misclassification(output)

        result = self.parent.run(self, model=model_correct_format, inputs=data.input, criterion=self.criterion, epsilon=self.epsilon)

        return result


class L2AdditiveGaussianNoise(L2AdditiveGaussianNoiseAttack, GenericAdditiveNoise):
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2AdditiveGaussianNoiseAttack, parameters)


class L2AdditiveUniformNoise(L2AdditiveUniformNoiseAttack, GenericAdditiveNoise):
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2AdditiveUniformNoiseAttack, parameters)


class L2ClippingAwareAdditiveGaussianNoise(L2ClippingAwareAdditiveGaussianNoiseAttack, GenericAdditiveNoise):
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2ClippingAwareAdditiveGaussianNoiseAttack, parameters)


class L2ClippingAwareAdditiveUniformNoise(L2ClippingAwareAdditiveUniformNoiseAttack, GenericAdditiveNoise):
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2ClippingAwareAdditiveUniformNoiseAttack, parameters)


class LinfAdditiveUniformNoise(LinfAdditiveUniformNoiseAttack, GenericAdditiveNoise):
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, LinfAdditiveUniformNoiseAttack, parameters)


class L2RepeatedAdditiveGaussianNoise(L2RepeatedAdditiveGaussianNoiseAttack, GenericAdditiveNoise):
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2RepeatedAdditiveGaussianNoiseAttack, parameters)


class L2RepeatedAdditiveUniformNoise(L2RepeatedAdditiveUniformNoiseAttack, GenericAdditiveNoise):
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2RepeatedAdditiveUniformNoiseAttack, parameters)


class L2ClippingAwareRepeatedAdditiveGaussianNoise(L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack, GenericAdditiveNoise):
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack, parameters)


class L2ClippingAwareRepeatedAdditiveUniformNoise(L2ClippingAwareRepeatedAdditiveUniformNoiseAttack, GenericAdditiveNoise):
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2ClippingAwareRepeatedAdditiveUniformNoiseAttack, parameters)


class LinfRepeatedAdditiveUniformNoise(LinfRepeatedAdditiveUniformNoiseAttack, GenericAdditiveNoise):
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, LinfRepeatedAdditiveUniformNoiseAttack, parameters)

