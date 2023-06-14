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
        super().verify_bounds(data=data)
        output = super().flatten_output(data)
        model_correct_format = super().reformat_model(model)

        import torch
        # output = torch.zeros(100)
        for el in output:
            el.append(el[0])
        output = torch.tensor(output, requires_grad=False, dtype=torch.long)
        # output = torch.reshape(output, (100, 1))
        # output = torch.cat((output, output), 1)
        from utils.logger import test_logger

        if self.criterion_type == "targeted_misclassification":
            test_logger.debug(f"L2CarliniWagner - targeted")
            self.criterion = TargetedMisclassification(output)
        elif self.criterion_type == "misclassification":
            test_logger.debug(f"L2CarliniWagner - NOT targeted - misclassification")
            self.criterion = Misclassification(output)
        else:
            test_logger.error("unknown classification")
            exit(1)

        import eagerpy as ep
        x = ep.astensor(data.input)
        test_logger.debug(f"L2CarliniWagner - {output.shape} {self.criterion.labels} {data.input}, {x}, {x.shape}")

        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)

        return result