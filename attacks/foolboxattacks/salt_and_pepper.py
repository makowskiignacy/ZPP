from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification
from foolbox.attacks.saltandpepper import SaltAndPepperNoiseAttack


class SaltAndPepperNoise(SaltAndPepperNoiseAttack, FoolboxAttack):
    def __init__(self, attack_specific_args, generic_args):

        super().__init__(**attack_specific_args)
        FoolboxAttack.__init__(self, generic_args)

    def conduct(self, model, data):

        model_correct_format = super().reformat_model(model)

        output = super().flatten_output(data)
        self.criterion = Misclassification(output)

        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)
        return result
