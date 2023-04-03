from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification
from foolbox.attacks.newtonfool import NewtonFoolAttack

from attacks.helpers.data import Data
from eagerpy.astensor import astensor

class NewtonFool(NewtonFoolAttack, FoolboxAttack):
    def __init__(self, attack_specific_args, generic_args):
        super().__init__(**attack_specific_args)
        FoolboxAttack.__init__(self, generic_args)

    def verify_bounds(self, data: Data):
        if hasattr(self, 'min') and hasattr(self, 'max'):
            print(f"#############################3 attr check: {hasattr(self, 'min')}, {hasattr(self, 'max')}, {self.min} {self.max}")

            return
        
        originals, _ = astensor(data.input)
        print(f"#########################3 bounds check: {originals.min().item()}, {originals.max().item()}")
        self.min = originals.min().item()
        self.max = originals.max().item()

    def conduct(self, model, data):
        self.verify_bounds(data=data)
        model_correct_format = super().reformat_model(model)

        if model_correct_format is None:
            model_correct_format = model
        output = super().flatten_output(data)
        self.criterion = Misclassification(output)
        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)
        # result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion, epsilon=self.epsilon)

        return result