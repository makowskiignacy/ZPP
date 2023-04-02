from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.saltandpepper import SaltAndPepperNoiseAttack


class SaltAndPepperNoise(SaltAndPepperNoiseAttack, FoolboxAttack):
    def __init__(self, attack_specific_args, generic_args):

        super().__init__(**attack_specific_args)
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

        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)
        return result
