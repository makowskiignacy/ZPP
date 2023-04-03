import traceback

from foolbox.attacks.brendel_bethge import L0BrendelBethgeAttack, L1BrendelBethgeAttack, L2BrendelBethgeAttack, LinfinityBrendelBethgeAttack

from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from abc import ABC

from attacks.helpers.data import Data
from eagerpy.astensor import astensor

class BrendelBethge(FoolboxAttack, ABC):
    def __init__(self, args):
        FoolboxAttack.__init__(self, args)
        self.attack = None

        if 'lr' in args:
            self.lr = args['lr']
        else:
            self.lr = 1

        if 'epsilons' in args:
            self.epsilons = args['epsilons']
        else:
            self.epsilons = 1

        if 'steps' in args:
            self.steps = args['steps']
        else:
            self.steps = 100


    def verify_bounds(self, data: Data):
        if hasattr(self, 'min') and hasattr(self, 'max'):
            return
        
        originals, _ = astensor(data.input)
        self.min = originals.min().item()
        self.max = originals.max().item()


    def conduct(self, model, data: Data):
        outputs = data.output
        self.verify_bounds(data=data)
        model_correct_format = super().reformat_model(model)

        if model_correct_format is None:
            model_correct_format = model
        else:
            outputs = data.output[:, 0]
        if self.criterion_type == "targeted_misclassification":
            self.criterion = TargetedMisclassification(outputs)
        if self.criterion_type == "misclassification":
            self.criterion = Misclassification(outputs)


        try:
            assert(self.attack is not None)

            adversarials, _, _ = self.attack(model=model_correct_format, inputs=data.input, criterion=self.criterion, epsilons=self.epsilons)


            return adversarials
        except Exception as e:
            delim = "\n" + "#" * 64 + "\n"
            print(f"Exception raised:{delim}{e}{delim}{repr(e)}{delim}{traceback.print_exc()}{delim}")
            return None
    

class L0BrendelBethge(BrendelBethge):
    def __init__(self, args):
        BrendelBethge.__init__(self, args)
        self.attack = L0BrendelBethgeAttack(lr=self.lr, steps=self.steps)


class L1BrendelBethge(BrendelBethge, L1BrendelBethgeAttack):
    def __init__(self, args):
        BrendelBethge.__init__(self, args)
        self.attack = L1BrendelBethgeAttack(lr=self.lr, steps=self.steps)


class L2BrendelBethge(BrendelBethge):
    def __init__(self, args):
        BrendelBethge.__init__(self, args)
        self.attack = L2BrendelBethgeAttack(lr=self.lr, steps=self.steps)


class LinfinityBrendelBethge(BrendelBethge, LinfinityBrendelBethgeAttack):
    def __init__(self, args):
        BrendelBethge.__init__(self, args)
        self.attack = LinfinityBrendelBethgeAttack(lr=self.lr, steps=self.steps)
