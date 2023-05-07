from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.carlini_wagner import L2CarliniWagnerAttack

from attacks.helpers.data import Data
from eagerpy.astensor import astensor


class L2CarliniWagner(L2CarliniWagnerAttack, FoolboxAttack):
    def __init__(self, parameters):
        super().__init__(**parameters.attack_specific_parameters)
        FoolboxAttack.__init__(self, parameters.generic_parameters)


    def conduct(self, model, data):
        super.verify_bounds(data=data)
        output = super().flatten_output(data)
        model_correct_format = super().reformat_model(model)

        if self.criterion_type == "targeted_misclassification":
            self.criterion = TargetedMisclassification(output)
        if self.criterion_type == "misclassification":
            self.criterion = Misclassification(output)

        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)

        return result