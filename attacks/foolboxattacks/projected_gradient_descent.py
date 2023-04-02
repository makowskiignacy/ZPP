from attacks.foolbox_attack import FoolboxAttack
from foolbox.attacks import L1PGD,L2PGD,LinfPGD,L1AdamPGD,L2AdamPGD,LinfAdamPGD
from foolbox.criteria import Misclassification, TargetedMisclassification


class GenericProjectedGradientDescent(FoolboxAttack):
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



class L1ProjectedGradientDescent(GenericProjectedGradientDescent, L1PGD):
    def __init__(self, attack_specific_args, generic_args):
        GenericProjectedGradientDescent.__init__(self, L1PGD, attack_specific_args, generic_args)


class L2ProjectedGradientDescent(GenericProjectedGradientDescent, L2PGD):
    def __init__(self, attack_specific_args, generic_args):
        GenericProjectedGradientDescent.__init__(self, L2PGD, attack_specific_args, generic_args)


class LinfProjectedGradientDescent(GenericProjectedGradientDescent, LinfPGD):
    def __init__(self, attack_specific_args, generic_args):
        GenericProjectedGradientDescent.__init__(self, LinfPGD, attack_specific_args, generic_args)


class L1AdamProjectedGradientDescent(GenericProjectedGradientDescent, L1AdamPGD):
    def __init__(self, attack_specific_args, generic_args):
        GenericProjectedGradientDescent.__init__(self, L1AdamPGD, attack_specific_args, generic_args)


class L2AdamProjectedGradientDescent(GenericProjectedGradientDescent, L2AdamPGD):
    def __init__(self, attack_specific_args, generic_args):
        GenericProjectedGradientDescent.__init__(self, L2AdamPGD, attack_specific_args, generic_args)


class LinfAdamProjectedGradientDescent(GenericProjectedGradientDescent, LinfAdamPGD):
    def __init__(self, attack_specific_args, generic_args):
        GenericProjectedGradientDescent.__init__(self, LinfAdamPGD, attack_specific_args, generic_args)
