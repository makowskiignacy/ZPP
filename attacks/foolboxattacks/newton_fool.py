from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification
from foolbox.attacks.newtonfool import NewtonFoolAttack

from attacks.helpers.data import Data
from eagerpy.astensor import astensor

import eagerpy as ep

class NewtonFool(NewtonFoolAttack, FoolboxAttack):
    def __init__(self, attack_specific_args, generic_args):
        super().__init__(**attack_specific_args)
        FoolboxAttack.__init__(self, generic_args)

    def verify_bounds(self, data: Data):
        if hasattr(self, 'min') and hasattr(self, 'max'):
            print(f"#############################3 attr check: {hasattr(self, 'min')}, {hasattr(self, 'max')}, {self.min}, {self.max}")

            return
        
        originals, _ = astensor(data.input)
        print(f"#########################3 bounds check: {originals.min().item()}, {originals.max().item()}")
        self.min = originals.min().item()
        self.max = originals.max().item()

        return

    def conduct(self, model, data):
        print(f"data type: {type(data)}, {type(data.input)}, {type(data.input[0][0])}")
        print(f"data in: {data.input}, {data.input[0]}")
        print(f"data out: {data.output}, {data.output[0]}")

        for o in data.output:
            if o.item() != 0.0 and o.item() != 1.0:
                print(o.item())

        self.verify_bounds(data=data)
        # model_correct_format = super().reformat_model(model)

        # if model_correct_format is None:
        #     model_correct_format = model
        model_correct_format = model


        output = super().flatten_output(data)
        self.criterion = Misclassification(output)


        x, restore_type = ep.astensor_(data.input)
        classes = self.criterion.labels
        print(f"classes.shape: {classes.shape}")
        logits = model(x)
        scores = ep.softmax(logits)
        print(f"scores: {scores}")
        # pred_scores = scores[0][range(len(x)), classes]
        # print(f"pred_scores: {pred_scores}")
        # # result = None
        print(f"criterion: {self.criterion}, {self.criterion.labels}, {len(self.criterion.labels)}, {type(self.criterion.labels[0])}")
        # print(f"inputs: x: {x}, restore_type: {restore_type}")
        # for l in self.criterion.labels:
        #     print(l)
        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)
        # result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion, epsilon=self.epsilon)

        return result