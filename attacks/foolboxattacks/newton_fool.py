from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.newtonfool import NewtonFoolAttack

from attacks.helpers.data import Data
from eagerpy.astensor import astensor

import eagerpy as ep
import torch
import numpy as np

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


    def check(self, model, data):
        x, restore_type = ep.astensor_(data.input)
        classes = self.criterion.labels
        print(f"classes.shape: {classes.shape}")
        logits = model(x)
        scores = ep.softmax(logits)
        # scores = scores.raw
        # print(f"scores: {type(scores[0])}, {len(scores)}, {scores[0]}")
        print(f"classes: {type(classes)}, {type(classes[0].item())}, {type(classes.raw.round().to(torch.int)[0].item())}, {classes[0]}")
        print(f"logits: {type(logits)}")
        
        # for s in scores.raw:
        #     if s.item() != 1.0:
        #         print(s.item())
        classes = classes.raw.round().to(torch.int)
        classes = [0] * len(x)
        pred_scores = scores[range(len(x)), classes]
        print(f"pred_scores: {pred_scores}")
        # # result = None
        print(f"criterion: {self.criterion}, {self.criterion.labels}, {len(self.criterion.labels)}, {type(self.criterion.labels[0])}")
        # print(f"inputs: x: {x}, restore_type: {restore_type}")
        # for l in self.criterion.labels:
        #     print(l)

    def prep_labels(self):
        print(type(self.criterion.labels))
        for l in self.criterion.labels:
            if l.item() != 0:
                print(l.item())


    def conduct(self, model, data):
        self.verify_bounds(data=data)
        # model_correct_format = super().reformat_model(model)

        # if model_correct_format is None:
        #     model_correct_format = model
        model_correct_format = model
        output = super().flatten_output(data)

        if self.criterion_type == "targeted_misclassification":
            print("targeted")
            self.criterion = TargetedMisclassification(output)
        if self.criterion_type == "misclassification":
            print("not targeted")
            self.criterion = Misclassification(output)

        # self.check(model, data)
        print(self.criterion.labels)
        self.prep_labels()
        # self.criterion.labels = self.prep_labels()
        # self.criterion.labels = torch.full((len(data.input),), 0)
        print(self.criterion.labels)

        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)

        return result