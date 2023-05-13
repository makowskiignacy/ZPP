from attacks.artattacks.fast_gradient import FastGradient
from attacks.artattacks.joker import Joker
from attacks.artattacks.jacobian_saliency_map import JacobianSaliencyMap
from attacks.artattacks.geometric_decision_based import GeometricDecisionBased
from attacks.artattacks.shadow import Shadow
from attacks.artattacks.threshold import Threshold
from attacks.artattacks.sign_opt import SignOPT
from attacks.artattacks.square import Square
from attacks.artattacks.zeroth_order_optimization_bb_attack import ZerothOrderOptimalization
from attacks.artattacks.deep_fool import DeepFool

from attacks.attack import run_attack
from utils.logger import test_logger


class ARTTest:
    def __init__(self, model, data, parameters):
        self.model = model
        self.data = data
        self.parameters = parameters


class TestFastGradient(ARTTest):
    def test(self):
        test_logger.info("Fast Gradient Attack test running")
        art_model = FastGradient(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestJoker(ARTTest):
    def test(self):
        test_logger.info("Joker Attack test running")
        art_model = Joker(joker='FastGradientMethod', parameters=self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestJacobianSaliencyMap(ARTTest):
    def test(self):
        test_logger.info("Jacobian Saliency Map Attack test running")
        art_model = JacobianSaliencyMap(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestGeometricDecisionBased(ARTTest):
    def test(self):
        test_logger.info("Geometric Decision Based Attack test running")
        art_model = GeometricDecisionBased(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestShadow(ARTTest):
    def test(self):
        test_logger.info("Shadow Attack test running")
        art_model = Shadow(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestThreshold(ARTTest):
    def test(self):
        test_logger.info("Threshold Attack test running")
        art_model = Threshold(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestSignOPT(ARTTest):
    def test(self):
        test_logger.info("SignOPT Attack test running")
        art_model = SignOPT(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestSquare(ARTTest):
    def test(self):
        test_logger.info("Square Attack test running")
        art_model = Square(self.parameters)
        return run_attack(art_model, self.model, self.data)


class TestZerothOrderOptimalization(ARTTest):
    def test(self):
        test_logger.info("Zeroth Order Optimalization test running")
        art_model = ZerothOrderOptimalization(self.parameters)
        return run_attack(art_model, self.model, self.data)

class TestDeepFool(ARTTest):
    def test(self):
        test_logger.info("Deep Fool test running")
        art_model = DeepFool(self.parameters)
        return run_attack(art_model, self.model, self.data)
