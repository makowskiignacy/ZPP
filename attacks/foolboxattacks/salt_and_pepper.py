from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.saltandpepper import SaltAndPepperNoiseAttack


class SaltAndPepperNoise(SaltAndPepperNoiseAttack, FoolboxAttack):
    def __init__(self, parameters):
        super().__init__(**parameters.attack_specific_parameters)
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

        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)
        return result
