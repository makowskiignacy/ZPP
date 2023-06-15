from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.basic_iterative_method import L1BasicIterativeAttack, L2BasicIterativeAttack, LinfBasicIterativeAttack
from foolbox.attacks.basic_iterative_method import L1AdamBasicIterativeAttack, L2AdamBasicIterativeAttack, LinfAdamBasicIterativeAttack

from utils.logger import test_logger


class GenericBasicIterative(FoolboxAttack):
    """
    :param parent - attack specific class from foolbox.attacks.basic_iterative_method module
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
        elif self.criterion_type == "misclassification":
            self.criterion = Misclassification(output)
        else:
            test_logger.error("unknown classification")
            exit(1)

        result = self.Parent.run(self, model=model_correct_format, inputs=data.input, criterion=self.criterion, epsilon=self.epsilon)
        return result


class L1BasicIterative(GenericBasicIterative, L1BasicIterativeAttack):
    def __init__(self, parameters):
        GenericBasicIterative.__init__(self, L1BasicIterativeAttack, parameters)


class L2BasicIterative(GenericBasicIterative, L2BasicIterativeAttack):
    def __init__(self, parameters):
        GenericBasicIterative.__init__(self, L2BasicIterativeAttack, parameters)


class LinfBasicIterative(GenericBasicIterative, LinfBasicIterativeAttack):
    def __init__(self, parameters):
        GenericBasicIterative.__init__(self, LinfBasicIterativeAttack, parameters)


class L1AdamBasicIterative(GenericBasicIterative, L1AdamBasicIterativeAttack):
    def __init__(self, parameters):
        GenericBasicIterative.__init__(self, L1AdamBasicIterativeAttack, parameters)


class L2AdamBasicIterative(GenericBasicIterative, L2AdamBasicIterativeAttack):
    def __init__(self, parameters):
        GenericBasicIterative.__init__(self, L2AdamBasicIterativeAttack, parameters)


class LinfAdamBasicIterative(GenericBasicIterative, LinfAdamBasicIterativeAttack):
    def __init__(self, parameters):
        GenericBasicIterative.__init__(self, LinfAdamBasicIterativeAttack, parameters)
