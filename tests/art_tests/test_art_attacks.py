from attacks.artattacks.fast_gradient import FastGradient
from attacks.artattacks.joker import Joker
from attacks.artattacks.jacobian_saliency_map import JacobianSaliencyMap
from attacks.artattacks.geometric_decision_based import GeometricDecisionBased
from attacks.artattacks.shadow import Shadow
from attacks.artattacks.threshold import Threshold
from attacks.artattacks.sign_opt import SignOPT
from attacks.artattacks.square import Square
from attacks.artattacks.zeroeth_order_optimization_bb_attack import ZeroethOrderOptimalization
from attacks.artattacks.deep_fool import DeepFool

from attacks.attack import run_attack


class ARTTest:
    def __init__(self, model, data, parameters):
        self.model = model
        self.data = data
        self.parameters = parameters


class TestFastGradient(ARTTest):
    def test(self):
        print("Fast Gradient Attack test running")
        art_model = FastGradient(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestJoker(ARTTest):
    def test(self):
        print("Joker Attack test running")
        art_model = Joker(joker='FastGradientMethod', parameters=self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestJacobianSaliencyMap(ARTTest):
    def test(self):
        print("Jacobian Saliency Map Attack test running")
        art_model = JacobianSaliencyMap(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestGeometricDecisionBased(ARTTest):
    def test(self):
        print("Geometric Decision Based Attack test running")
        art_model = GeometricDecisionBased(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestShadow(ARTTest):
    def test(self):
        print("Shadow Attack test running")
        art_model = Shadow(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestThreshold(ARTTest):
    def test(self):
        print("Threshold Attack test running")
        art_model = Threshold(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestSignOPT(ARTTest):
    def test(self):
        print("SignOPT Attack test running")
        art_model = SignOPT(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestSquare(ARTTest):
    def test(self):
        print("Square Attack test running")
        art_model = Square(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestZeroethOrderOptimalization(ARTTest):
    def test(self):
        print("Zeroeth Order Optimalization test running")
        art_model = ZeroethOrderOptimalization(self.parameters)
        return run_attack(art_model, self.model, self.data)

class TestDeepFool(ARTTest):
    def test(self):
        print("Deep Fool test running")
        art_model = DeepFool(self.parameters)
        return run_attack(art_model, self.model, self.data)
