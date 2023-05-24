from attacks.foolbox_attack import FoolboxAttack
from foolbox.attacks import L1PGD,L2PGD,LinfPGD,L1AdamPGD,L2AdamPGD,LinfAdamPGD
from foolbox.criteria import Misclassification, TargetedMisclassification
from attacks.helpers.data import Data
from eagerpy.astensor import astensor


class GenericProjectedGradientDescent(FoolboxAttack):
    """
    :param parent - attack specific class from foolbox.attacks module
    """
    def __init__(self, parent, parameters):
        self.Parent = parent
        self.Parent.__init__(self, **parameters.attack_specific_parameters)
        FoolboxAttack.__init__(self, parameters.generic_parameters)

    def conduct(self, model, data):

        output = super().flatten_output(data)
        super().verify_bounds(data=data)
        super().verify_epsilon()
        model_correct_format = super().reformat_model(model)

        if self.criterion_type == "targeted_misclassification":
            self.criterion = TargetedMisclassification(output)
        if self.criterion_type == "misclassification":
            self.criterion = Misclassification(output)

        result = self.Parent.run(self, model=model_correct_format, inputs=data.input, criterion=self.criterion, epsilon=self.epsilon)
        return result



class L1ProjectedGradientDescent(GenericProjectedGradientDescent, L1PGD):
    def __init__(self, parameters):
        GenericProjectedGradientDescent.__init__(self, L1PGD, parameters)


class L2ProjectedGradientDescent(GenericProjectedGradientDescent, L2PGD):
    def __init__(self, parameters):
        GenericProjectedGradientDescent.__init__(self, L2PGD, parameters)


class LinfProjectedGradientDescent(GenericProjectedGradientDescent, LinfPGD):
    def __init__(self, parameters):
        GenericProjectedGradientDescent.__init__(self, LinfPGD, parameters)


class L1AdamProjectedGradientDescent(GenericProjectedGradientDescent, L1AdamPGD):
    def __init__(self, parameters):
        GenericProjectedGradientDescent.__init__(self, L1AdamPGD, parameters)


class L2AdamProjectedGradientDescent(GenericProjectedGradientDescent, L2AdamPGD):
    def __init__(self, parameters):
        GenericProjectedGradientDescent.__init__(self, L2AdamPGD, parameters)


class LinfAdamProjectedGradientDescent(GenericProjectedGradientDescent, LinfAdamPGD):
    def __init__(self, parameters):
        GenericProjectedGradientDescent.__init__(self, LinfAdamPGD, parameters)
