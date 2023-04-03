from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.basic_iterative_method import L1BasicIterativeAttack, L2BasicIterativeAttack, LinfBasicIterativeAttack
from foolbox.attacks.basic_iterative_method import L1AdamBasicIterativeAttack, L2AdamBasicIterativeAttack, LinfAdamBasicIterativeAttack


class GenericBasicIterative(FoolboxAttack):
    def __init__(self, parent, attack_specific_args, generic_args):
        self.Parent = parent
        self.Parent.__init__(self, **attack_specific_args)
        FoolboxAttack.__init__(self, generic_args)

    def conduct(self, model, data):

        model_correct_format = super().reformat_model(model)
        if model_correct_format is None:
            model_correct_format = model

        output = super().flatten_output(data)
        if self.criterion_type == "targeted_misclassification":
            self.criterion = TargetedMisclassification(output)
        if self.criterion_type == "misclassification":
            self.criterion = Misclassification(output)

        result = self.Parent.run(self, model=model_correct_format, inputs=data.input, criterion=self.criterion, epsilon=self.epsilon)
        return result


class L1BasicIterative(GenericBasicIterative, L1BasicIterativeAttack):
    def __init__(self, attack_specific_args, generic_args):
        GenericBasicIterative.__init__(self, L1BasicIterativeAttack, attack_specific_args, generic_args)


class L2BasicIterative(GenericBasicIterative, L2BasicIterativeAttack):
    def __init__(self, attack_specific_args, generic_args):
        GenericBasicIterative.__init__(self, L2BasicIterativeAttack, attack_specific_args, generic_args)


class LinfBasicIterative(GenericBasicIterative, LinfBasicIterativeAttack):
    def __init__(self, attack_specific_args, generic_args):
        GenericBasicIterative.__init__(self, LinfBasicIterativeAttack, attack_specific_args, generic_args)


class L1AdamBasicIterative(GenericBasicIterative, L1AdamBasicIterativeAttack):
    def __init__(self, attack_specific_args, generic_args):
        GenericBasicIterative.__init__(self, L1AdamBasicIterativeAttack, attack_specific_args, generic_args)


class L2AdamBasicIterative(GenericBasicIterative, L2AdamBasicIterativeAttack):
    def __init__(self, attack_specific_args, generic_args):
        GenericBasicIterative.__init__(self, L2AdamBasicIterativeAttack, attack_specific_args, generic_args)


class LinfAdamBasicIterative(GenericBasicIterative, LinfAdamBasicIterativeAttack):
    def __init__(self, attack_specific_args, generic_args):
        GenericBasicIterative.__init__(self, LinfAdamBasicIterativeAttack, attack_specific_args, generic_args)
