import traceback

from foolbox.attacks.brendel_bethge import L0BrendelBethgeAttack, L1BrendelBethgeAttack, L2BrendelBethgeAttack, LinfinityBrendelBethgeAttack

from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from abc import ABC


class GenericBrendelBethge(FoolboxAttack, ABC):
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

        result = self.Parent.run(self, model=model_correct_format, inputs=data.input, criterion=self.criterion)
        return result



class L0BrendelBethge(GenericBrendelBethge, L0BrendelBethgeAttack):
    def __init__(self, parameters):
        GenericBrendelBethge.__init__(self, L0BrendelBethgeAttack, parameters)


class L1BrendelBethge(GenericBrendelBethge, L1BrendelBethgeAttack):
    def __init__(self, parameters):
        GenericBrendelBethge.__init__(self, L1BrendelBethgeAttack, parameters)


class L2BrendelBethge(GenericBrendelBethge, L2BrendelBethgeAttack):
    def __init__(self, parameters):
        GenericBrendelBethge.__init__(self, L2BrendelBethgeAttack, parameters)


class LinfBrendelBethge(GenericBrendelBethge, LinfinityBrendelBethgeAttack):
    def __init__(self, parameters):
        GenericBrendelBethge.__init__(self, LinfinityBrendelBethgeAttack, parameters)
