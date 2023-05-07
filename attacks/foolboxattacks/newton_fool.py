from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.newtonfool import NewtonFoolAttack

from attacks.helpers.data import Data
from eagerpy.astensor import astensor


class NewtonFool(NewtonFoolAttack, FoolboxAttack):
    def __init__(self, parameters):
        super().__init__(**parameters.attack_specific_parameters)
        FoolboxAttack.__init__(self, parameters.generic_parameters)

    def verify_bounds(self, data: Data):
        if hasattr(self, 'min') and hasattr(self, 'max'):
            return
        
        originals, _ = astensor(data.input)
        self.min = originals.min().item()
        self.max = originals.max().item()

        return


    def conduct(self, model, data):
        self.verify_bounds(data=data)
        output = super().flatten_output(data)
        model_correct_format = super().reformat_model(model)

        if self.criterion_type == "targeted_misclassification":
            self.criterion = TargetedMisclassification(output)
        if self.criterion_type == "misclassification":
            self.criterion = Misclassification(output)

        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)

        return result