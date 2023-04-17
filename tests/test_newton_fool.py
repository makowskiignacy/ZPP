from attacks.foolboxattacks.newton_fool import NewtonFool
from tests.test_class import Test
# from tests import Test
import unittest

class NewtonFool(unittest.TestCase):
    generic_args = {}
    attack_specific_args_simple = {'steps': 100}
    attack_specific_args_nn = {'steps': 100}

    attack_simple = NewtonFool(attack_specific_args_simple, generic_args)
    attack_nn = NewtonFool(attack_specific_args_nn, generic_args)

    testor = Test(attack_simple=attack_simple, attack_nn=attack_nn, batchsize=4)


    # def test_nf_simple(self):
    #     results = self.testor.test_simple()
    #     self.assertIsNotNone(results)


    def test_nf_nn(self):
        results = self.testor.test_nn()
        self.assertIsNotNone(results)
