import math
import sys

from foolbox.attacks import LinearSearchBlendedUniformNoiseAttack
from foolbox.attacks.brendel_bethge import BrendelBethgeAttack, L0BrendelBethgeAttack, L1BrendelBethgeAttack, L2BrendelBethgeAttack, LinfinityBrendelBethgeAttack
from foolbox.distances import LpDistance

from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification
import eagerpy as ep


class L0BrendelBethge(FoolboxAttack, L0BrendelBethgeAttack):
    """L1 variant of the Brendel & Bethge adversarial attack. [#Bren19]_
    This is a powerful gradient-based adversarial attack that follows the
    adversarial boundary (the boundary between the space of adversarial and
    non-adversarial images as defined by the adversarial criterion) to find
    the minimum distance to the clean image.
    This is the reference implementation of the Brendel & Bethge attack.
    References:
        .. [#Bren19] Wieland Brendel, Jonas Rauber, Matthias Kümmerer,
           Ivan Ustyuzhaninov, Matthias Bethge,
           "Accurate, reliable and fast robustness evaluation",
           33rd Conference on Neural Information Processing Systems (2019)
           https://arxiv.org/abs/1907.01003
    """

    def __init__(self, args):
        # super().__init__(args)
        FoolboxAttack.__init__(self, args)
        BrendelBethgeAttack.__init__(self)

        # FoolboxAttack.__init__(self, args)
    def generate_starting_points(self, data, model, distance, directions=1000, steps=1000):
        # default params for LinearSearchBlendedUniformNoiseAttack: distance=None, directions=1000, steps=1000
        init_attack = LinearSearchBlendedUniformNoiseAttack(distance=distance, directions=directions, steps=steps)
        originals, _ = ep.astensor_(data.input)
        starting_points = init_attack.run(model, originals, self.criterion)

        if starting_points is None:
            print(f'Wrong starting points ({starting_points}) for params: distance={distance}, directions={directions}, steps={steps}')
        else:
            print(f'Successful starting points ({starting_points}) for params: distance={distance}, directions={directions}, steps={steps}')
            # sys.exit(f'Wrong starting points ({starting_points}) for params: distance={dist}, directions={dirs}, steps={stps}')

        return starting_points

    def generate_starting_points_set(self, data, model):
        dist = LpDistance(p=2)
        directions2check = [10, 100, 1000, 10000, 100000, 1000000]
        steps2check = [10, 100, 1000, 10000, 100000, 1000000]
        distances2check = [LpDistance(0), LpDistance(1), LpDistance(2), LpDistance(math.inf)]
        ptsSet = []
        # stps = 100000
        # default params for LinearSearchBlendedUniformNoiseAttack: distance=None, directions=1000, steps=1000
        for dist in distances2check:
            for dir in directions2check:
                for step in steps2check:
                    init_attack = LinearSearchBlendedUniformNoiseAttack(distance=dist, directions=dir, steps=step)
                    originals, _ = ep.astensor_(data.input)
                    starting_points = init_attack.run(model, originals, self.criterion)

                    if starting_points is None:
                        print(f'Wrong starting points ({starting_points}) for params: distance={dist}, directions={dir}, steps={step}')
                    else:
                        print(f'Successful starting points ({starting_points}) for params: distance={dist}, directions={dir}, steps={step}')
                        # sys.exit(f'Wrong starting points ({starting_points}) for params: distance={dist}, directions={dirs}, steps={stps}')
                        ptsSet.append(starting_points)

        return ptsSet

    def conduct(self, model, data):
        model_correct_format = super().reformat_model(model)
        outputs = data.output[:, 0]
        self.criterion = Misclassification(outputs)

        print("generating starting_points")
        # starting_points = self.generate_starting_points(data, model_correct_format, 0)
        starting_points = self.generate_starting_points_set(data, model_correct_format)
        print("starting_points done -> attack")

        # result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)
        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion, starting_points=starting_points[0])

        return result


class L1BrendelBethge(FoolboxAttack, L1BrendelBethgeAttack):
    """L1 variant of the Brendel & Bethge adversarial attack. [#Bren19]_
    This is a powerful gradient-based adversarial attack that follows the
    adversarial boundary (the boundary between the space of adversarial and
    non-adversarial images as defined by the adversarial criterion) to find
    the minimum distance to the clean image.
    This is the reference implementation of the Brendel & Bethge attack.
    References:
        .. [#Bren19] Wieland Brendel, Jonas Rauber, Matthias Kümmerer,
           Ivan Ustyuzhaninov, Matthias Bethge,
           "Accurate, reliable and fast robustness evaluation",
           33rd Conference on Neural Information Processing Systems (2019)
           https://arxiv.org/abs/1907.01003
    """

    def __init__(self, args):
        # super().__init__(args)
        FoolboxAttack.__init__(self, args)
        BrendelBethgeAttack.__init__(self)

        # FoolboxAttack.__init__(self, args)

    def conduct(self, model, data):
        model_correct_format = super().reformat_model(model)
        outputs = data.output[:, 0]
        self.criterion = Misclassification(outputs)
        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)

        return result


class L2BrendelBethge(FoolboxAttack, L2BrendelBethgeAttack):
    """L1 variant of the Brendel & Bethge adversarial attack. [#Bren19]_
    This is a powerful gradient-based adversarial attack that follows the
    adversarial boundary (the boundary between the space of adversarial and
    non-adversarial images as defined by the adversarial criterion) to find
    the minimum distance to the clean image.
    This is the reference implementation of the Brendel & Bethge attack.
    References:
        .. [#Bren19] Wieland Brendel, Jonas Rauber, Matthias Kümmerer,
           Ivan Ustyuzhaninov, Matthias Bethge,
           "Accurate, reliable and fast robustness evaluation",
           33rd Conference on Neural Information Processing Systems (2019)
           https://arxiv.org/abs/1907.01003
    """

    def __init__(self, args):
        # super().__init__(args)
        FoolboxAttack.__init__(self, args)
        BrendelBethgeAttack.__init__(self)

        # FoolboxAttack.__init__(self, args)

    def conduct(self, model, data):
        model_correct_format = super().reformat_model(model)
        outputs = data.output[:, 0]
        self.criterion = Misclassification(outputs)
        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)

        return result


class LinfinityBrendelBethge(FoolboxAttack, LinfinityBrendelBethgeAttack):
    """L1 variant of the Brendel & Bethge adversarial attack. [#Bren19]_
    This is a powerful gradient-based adversarial attack that follows the
    adversarial boundary (the boundary between the space of adversarial and
    non-adversarial images as defined by the adversarial criterion) to find
    the minimum distance to the clean image.
    This is the reference implementation of the Brendel & Bethge attack.
    References:
        .. [#Bren19] Wieland Brendel, Jonas Rauber, Matthias Kümmerer,
           Ivan Ustyuzhaninov, Matthias Bethge,
           "Accurate, reliable and fast robustness evaluation",
           33rd Conference on Neural Information Processing Systems (2019)
           https://arxiv.org/abs/1907.01003
    """

    def __init__(self, args):
        # super().__init__(args)
        FoolboxAttack.__init__(self, args)
        BrendelBethgeAttack.__init__(self)

        # FoolboxAttack.__init__(self, args)

    def conduct(self, model, data):
        model_correct_format = super().reformat_model(model)
        outputs = data.output[:, 0]
        self.criterion = Misclassification(outputs)
        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)

        return result