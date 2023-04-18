from attacks.foolboxattacks.additive_noise import L2AdditiveGaussianNoise, L2AdditiveUniformNoise, L2ClippingAwareAdditiveGaussianNoise, L2ClippingAwareAdditiveUniformNoise, LinfAdditiveUniformNoise, L2RepeatedAdditiveGaussianNoise, L2RepeatedAdditiveUniformNoise, L2ClippingAwareRepeatedAdditiveGaussianNoise, L2ClippingAwareRepeatedAdditiveUniformNoise, LinfRepeatedAdditiveUniformNoise
from attacks.helpers.parameters import FoolboxParameters
from tests.tester_class import Test
import unittest


def test_msg(test_name):
    print(f"\n### {test_name} L2AdditiveGaussianNoise test: ###")

class TestL2AdditiveGaussianNoise(unittest.TestCase):
    generic_args = {'epsilon': 1}
    attack_specific_args = {}

    parameters_simple = FoolboxParameters(attack_specific_args, generic_args)
    parameters_nn = FoolboxParameters(attack_specific_args, generic_args)

    attack_simple = L2AdditiveGaussianNoise(parameters_simple)
    attack_nn = L2AdditiveGaussianNoise(parameters_nn)

    tester = Test(attack_simple=attack_simple, attack_nn=attack_nn, batchsize=4)


    # def test_nf_simple(self):
    #     test_msg("simple")
    #     results = self.tester.test_simple()
    #     self.assertIsNotNone(results)


    def test_nf_nn(self):
        test_msg("nn")
        results = self.tester.test_nn()
        self.assertIsNotNone(results)


class TestAdditiveNoise(unittest.TestCase):
    an_params = {
        'L2AdditiveGaussianNoise': {'attack': L2AdditiveGaussianNoise, 'params': {'epsilon': 0.001}}, 
        'L2AdditiveUniformNoise': {'attack': L2AdditiveUniformNoise, 'params': {'epsilon': 0.001}}, 
        'L2ClippingAwareAdditiveGaussianNoise': {'attack': L2ClippingAwareAdditiveGaussianNoise, 'params': {'epsilon': 0.001}}, 
        'L2ClippingAwareAdditiveUniformNoise': {'attack': L2ClippingAwareAdditiveUniformNoise, 'params': {'epsilon': 0.001}}, 
        'LinfAdditiveUniformNoise': {'attack': LinfAdditiveUniformNoise, 'params': {'epsilon': 0.001}}, 
        'L2RepeatedAdditiveGaussianNoise': {'attack': L2RepeatedAdditiveGaussianNoise, 'params': {'epsilon': 0.001}}, 
        'L2RepeatedAdditiveUniformNoise': {'attack': L2RepeatedAdditiveUniformNoise, 'params': {'epsilon': 0.001}}, 
        'L2ClippingAwareRepeatedAdditiveGaussianNoise': {'attack': L2ClippingAwareRepeatedAdditiveGaussianNoise, 'params': {'epsilon': 0.001}}, 
        'L2ClippingAwareRepeatedAdditiveUniformNoise': {'attack': L2ClippingAwareRepeatedAdditiveUniformNoise, 'params': {'epsilon': 0.001}}, 
        'LinfRepeatedAdditiveUniformNoise': {'attack': LinfRepeatedAdditiveUniformNoise, 'params': {'epsilon': 0.001}},
    }

    generic_args = {'epsilon': 1}
    attack_specific_args = {}

    parameters_simple = FoolboxParameters(attack_specific_args, generic_args)
    parameters_nn = FoolboxParameters(attack_specific_args, generic_args)

    attack_simple = L2AdditiveGaussianNoise(parameters_simple)
    attack_nn = L2AdditiveGaussianNoise(parameters_nn)

    tester = Test(attack_simple=attack_simple, attack_nn=attack_nn, batchsize=4)


    # def test_nf_simple(self):
    #     test_msg("simple")
    #     results = self.tester.test_simple()
    #     self.assertIsNotNone(results)


    def test_test(self):
        for key, val in self.an_params.items():
            print(key)
            attack_simple = val.get('attack')(self.parameters_simple)
            attack_nn = val.get('attack')(self.parameters_nn)

            tester = Test(attack_simple=attack_simple, attack_nn=attack_nn, batchsize=4)

            res = tester.test_nn()
            self.assertIsNotNone(res)


    # def test_nf_nn(self):
    #     test_msg("nn")
    #     results = self.tester.test_nn()
    #     self.assertIsNotNone(results)
