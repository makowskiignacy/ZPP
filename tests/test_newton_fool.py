from attacks.foolboxattacks.newton_fool import NewtonFool
from attacks.helpers.parameters import FoolboxParameters
from tests.tester_class import Test
import unittest

def test_msg(test_name):
    print(f"\n### {test_name} NewtonFool test: ###")

class TestNewtonFool(unittest.TestCase):
    generic_args = {}
    attack_specific_args = {}

    parameters_simple = FoolboxParameters(attack_specific_args, generic_args)
    parameters_nn = FoolboxParameters(attack_specific_args, generic_args)

    attack_simple = NewtonFool(parameters_simple)
    attack_nn = NewtonFool(parameters_nn)

    tester = Test(attack_simple=attack_simple, attack_nn=attack_nn, batchsize=4)


    def test_nf_simple(self):
        test_msg("simple")
        results = self.tester.test_simple()
        self.assertIsNotNone(results)


    def test_nf_nn(self):
        test_msg("nn")
        results = self.tester.test_nn()
        self.assertIsNotNone(results)
