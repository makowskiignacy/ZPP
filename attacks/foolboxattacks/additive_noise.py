from abc import ABC
from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.additive_noise import L2AdditiveGaussianNoiseAttack, L2AdditiveUniformNoiseAttack, L2ClippingAwareAdditiveGaussianNoiseAttack, L2ClippingAwareAdditiveUniformNoiseAttack, LinfAdditiveUniformNoiseAttack, L2RepeatedAdditiveGaussianNoiseAttack, L2RepeatedAdditiveUniformNoiseAttack, L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack, L2ClippingAwareRepeatedAdditiveUniformNoiseAttack, LinfRepeatedAdditiveUniformNoiseAttack



from attacks.helpers.data import Data
from eagerpy.astensor import astensor


class GenericAdditiveNoise(FoolboxAttack, ABC):
    """
    Klasa generyczna dla ataku typu Additive Noise z biblioteki Foolbox
    """
    def __init__(self, parent, parameters):
        '''
        Inicjalizuje obiekt na podstawie wybranego rodzaju ataku Additive Noise

        Parametry:
        ----------
        parent (foolbox.attacks.additive_noise)
            Rodzaj ataku z rodziny Additive Noise do zainicjalizowania
        parameters
            Parametry odpowiednie dla wybranego ataku
        '''
        self.parent = parent
        self.parent.__init__(self, **parameters.attack_specific_parameters)
        FoolboxAttack.__init__(self, parameters.generic_parameters)



    def conduct(self, model, data):
        super().verify_bounds(data=data)
        super().verify_epsilon()
        output = super().flatten_output(data)
        model_correct_format = super().reformat_model(model)

        if self.criterion_type == "targeted_misclassification":
            self.criterion = TargetedMisclassification(output)
        if self.criterion_type == "misclassification":
            self.criterion = Misclassification(output)

        result = self.parent.run(self, model=model_correct_format, inputs=data.input, criterion=self.criterion, epsilon=self.epsilon)

        return result


class L2AdditiveGaussianNoise(L2AdditiveGaussianNoiseAttack, GenericAdditiveNoise):
    '''
    Atak próbkujący szum z rozkładu normalnego ze stałą normą L2
    '''
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2AdditiveGaussianNoiseAttack, parameters)


class L2AdditiveUniformNoise(L2AdditiveUniformNoiseAttack, GenericAdditiveNoise):
    '''
    Atak próbkujący szum z rozkładu jednostajnego ze stałą normą L2
    '''
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2AdditiveUniformNoiseAttack, parameters)


class L2ClippingAwareAdditiveGaussianNoise(L2ClippingAwareAdditiveGaussianNoiseAttack, GenericAdditiveNoise):
    '''
    Atak próbkujący szum z rozkładu normalnego ze stałą normą L2 po przycięciu
    wartości cech.

    Na podstawie: https://arxiv.org/abs/2007.07677
    '''
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2ClippingAwareAdditiveGaussianNoiseAttack, parameters)


class L2ClippingAwareAdditiveUniformNoise(L2ClippingAwareAdditiveUniformNoiseAttack, GenericAdditiveNoise):
    '''
    Atak próbkujący szum z rozkładu jednostajnego ze stałą normą L2 po
    przycięciu wartości cech (clipping).

    Na podstawie: https://arxiv.org/abs/2007.07677
    '''
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2ClippingAwareAdditiveUniformNoiseAttack, parameters)


class L2RepeatedAdditiveGaussianNoise(L2RepeatedAdditiveGaussianNoiseAttack, GenericAdditiveNoise):
    '''
    Atak próbkujący wielokrotnie szum z rozkładu normalnego ze stałą normą L2
    '''
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2RepeatedAdditiveGaussianNoiseAttack, parameters)


class L2RepeatedAdditiveUniformNoise(L2RepeatedAdditiveUniformNoiseAttack, GenericAdditiveNoise):
    '''
    Atak próbkujący wielokrotnie szum z rozkładu jednostajnego ze stałą normą L2
    '''
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2RepeatedAdditiveUniformNoiseAttack, parameters)


class L2ClippingAwareRepeatedAdditiveGaussianNoise(L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack, GenericAdditiveNoise):
    '''
    Atak próbkujący wielokrotnie szum z rozkładu normalnego
    ze stałą normą L2 po przycięciu wartości cech (clipping).

    Na podstawie: https://arxiv.org/abs/2007.07677
    '''
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack, parameters)


class L2ClippingAwareRepeatedAdditiveUniformNoise(L2ClippingAwareRepeatedAdditiveUniformNoiseAttack, GenericAdditiveNoise):
    '''
    Atak próbkujący wielokrotnie szum z rozkładu jednostajnego ze stałą normą L2
    po przycięciu wartości cech (clipping).

    Na podstawie: https://arxiv.org/abs/2007.07677
    '''
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, L2ClippingAwareRepeatedAdditiveUniformNoiseAttack, parameters)


class LinfAdditiveUniformNoise(LinfAdditiveUniformNoiseAttack, GenericAdditiveNoise):
    '''
    Atak próbkujący szum z rozkładu jednostajnego z stałą normą L-nieskończoność
    '''
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, LinfAdditiveUniformNoiseAttack, parameters)


class LinfRepeatedAdditiveUniformNoise(LinfRepeatedAdditiveUniformNoiseAttack, GenericAdditiveNoise):
    '''
    Atak próbkujący wielokrotnie szum z rozkładu jednostajnego z stałą normą
    L-nieskończoność
    '''
    def __init__(self, parameters):
        GenericAdditiveNoise.__init__(self, LinfRepeatedAdditiveUniformNoiseAttack, parameters)

