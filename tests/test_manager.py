import unittest
from utils.logger import *

from tests import input_loader

from tests.foolbox_tests.test_additive_noise import TestAdditiveNoise
from tests.foolbox_tests.test_brendel_bethge import TestBrendelBethge
from tests.foolbox_tests.test_basic_iterative import TestBasicIterative
from tests.foolbox_tests.test_carlini_wagner import TestCarliniWagner
from tests.foolbox_tests.test_newton_fool import TestNewtonFool
from tests.foolbox_tests.test_projected_gradient_descent import TestProjectedGradientDescent
from tests.foolbox_tests.test_salt_and_pepper import TestSaltAndPepper

from tests.art_tests.test_class_hierarchy_art import TestClassHierarchyArt
from tests.art_tests.test_art_attacks import *
from tests.other_tests.test_frameworks import *

#To run the tests on a different data set, add it to input_loader.py, and then change the line below.

# foolbox_model, art_model, foolbox_data, art_data, foolbox_parameters, art_parameters = input_loader.simple_input(batchsize = 20)
# foolbox_model, art_model, foolbox_data, art_data, foolbox_parameters, art_parameters = input_loader.nn_input()
# foolbox_model, art_model, foolbox_data, art_data, foolbox_parameters, art_parameters = input_loader.nn_input_cloud()
foolbox_model, art_model, foolbox_data, art_data, foolbox_parameters, art_parameters = input_loader.resnet18_cifar100_input(batchsize = 15)


class FoolboxTests(unittest.TestCase):

    def test_AdditiveNoise(self):
        attack_name = "Additive Noise"
        log_attack_start_msg(attack_name)
        parameters = foolbox_parameters.get("additive_noise")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestAdditiveNoise(foolbox_model, foolbox_data, parameters)
        self.assertIsNotNone(test.test_an_l2_g())
        self.assertIsNotNone(test.test_an_l2_u())
        self.assertIsNotNone(test.test_an_l2_cag())
        self.assertIsNotNone(test.test_an_l2_cau())
        self.assertIsNotNone(test.test_an_l2_carg())
        self.assertIsNotNone(test.test_an_l2_caru())
        self.assertIsNotNone(test.test_an_l2_rg())
        self.assertIsNotNone(test.test_an_l2_ru())
        self.assertIsNotNone(test.test_an_inf_u())
        self.assertIsNotNone(test.test_an_inf_ru())

    def test_BrendelBethge(self):
        attack_name = "Brendel-Bethge"
        log_attack_start_msg(attack_name)
        parameters = foolbox_parameters.get("brendel_bethge")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestBrendelBethge(foolbox_model, foolbox_data, parameters)
        self.assertIsNotNone(test.test_bb_0())
        self.assertIsNotNone(test.test_bb_1())
        self.assertIsNotNone(test.test_bb_2())
        self.assertIsNotNone(test.test_bb_inf())

    def test_BasicIterative(self):
        attack_name = "Basic Iterative"
        log_attack_start_msg(attack_name)
        parameters = foolbox_parameters.get("basic_iterative")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestBasicIterative(foolbox_model, foolbox_data, parameters)
        self.assertIsNotNone(test.test_bi_1())
        self.assertIsNotNone(test.test_bi_1_a())
        self.assertIsNotNone(test.test_bi_2())
        self.assertIsNotNone(test.test_bi_2_a())
        self.assertIsNotNone(test.test_bi_inf())
        self.assertIsNotNone(test.test_bi_inf_a())

    def test_CarliniWagner(self):
        attack_name = "Carlini Wagner"
        log_attack_start_msg(attack_name)
        parameters = foolbox_parameters.get("carlini_wagner")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestCarliniWagner(foolbox_model, foolbox_data, parameters)
        self.assertIsNotNone(test.test_cw_l2())

    def test_NewtonFool(self):
        attack_name = "Newton Fool"
        log_attack_start_msg(attack_name)
        parameters = foolbox_parameters.get("newton_fool")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestNewtonFool(foolbox_model, foolbox_data, parameters)
        self.assertIsNotNone(test.test_nf())

    def test_ProjectedGradientDescent(self):
        attack_name = "Projected Gradient"
        log_attack_start_msg(attack_name)
        parameters = foolbox_parameters.get("projected_gradient_descent")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestProjectedGradientDescent(foolbox_model, foolbox_data, parameters)
        self.assertIsNotNone(test.test_pgd_1())
        self.assertIsNotNone(test.test_pgd_1_a())
        self.assertIsNotNone(test.test_pgd_2())
        self.assertIsNotNone(test.test_pgd_2_a())
        self.assertIsNotNone(test.test_pgd_inf())
        self.assertIsNotNone(test.test_pgd_inf_a())

    def test_SaltAndPepperNoise(self):
        attack_name = "Salt and Pepper"
        log_attack_start_msg(attack_name)
        parameters = foolbox_parameters.get("salt_and_pepper")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestSaltAndPepper(foolbox_model, foolbox_data, parameters)
        self.assertIsNotNone(test.test_sap())


class ArtTests(unittest.TestCase):

    def test_ClassHierarchy(self):
        test = TestClassHierarchyArt()
        self.assertEqual(test.test_class_hierarchy(), [True, True, True])

    def test_DeepFool(self):
        attack_name = "Deep Fool"
        log_attack_start_msg(attack_name)
        parameters = art_parameters.get("deep_fool")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestDeepFool(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_FastGradient(self):
        attack_name = "Fast Gradient"
        log_attack_start_msg(attack_name)
        parameters = art_parameters.get("fast_gradient")
        self.assertIsNotNone(parameters,msg=input_err_msg(attack_name))
        test = TestFastGradient(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_Joker(self):
        attack_name = "Joker"
        log_attack_start_msg(attack_name)
        parameters = art_parameters.get("joker")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestJoker(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_JacobianSaliencyMap(self):
        attack_name = "Jacobian Saliency"
        log_attack_start_msg(attack_name)
        parameters = art_parameters.get("jacobian_saliency_map")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestJacobianSaliencyMap(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_GeometricDecisionBased(self):
        attack_name = "Geometric Decision"
        log_attack_start_msg(attack_name)
        parameters = art_parameters.get("geometric_decision_based")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestGeometricDecisionBased(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_Shadow(self):
        attack_name = "Shadow"
        log_attack_start_msg(attack_name)
        parameters = art_parameters.get("shadow")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestShadow(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_Threshold(self):
        attack_name = "Threshold"
        log_attack_start_msg(attack_name)
        parameters = art_parameters.get("threshold")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestThreshold(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_SignOPT(self):
        attack_name = "Sign-OPT"
        log_attack_start_msg(attack_name)
        parameters = art_parameters.get("sign_opt")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestSignOPT(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_Square(self):
        attack_name = "Square"
        log_attack_start_msg(attack_name)
        parameters = art_parameters.get("square")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestSquare(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_ZerothOrderOptimalization(self):
        attack_name = "Zeroth Order"
        log_attack_start_msg(attack_name)
        parameters = art_parameters.get("zeroth_order_optimization")
        self.assertIsNotNone(parameters, msg=input_err_msg(attack_name))
        test = TestZerothOrderOptimalization(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

class TestFrameworks(unittest.TestCase):

    def test_FoolboxWithPytorchUsingFoolbox(self):
        test = TestFoolboxWithPytorchUsingFoolbox()
        self.assertIsNotNone(test.test_foolbox_L1BasicIterative())
        self.assertIsNotNone(test.test_foolbox_ProjectedGradientDescentInf())

    def test_FoolboxWithPytorchUsingArt(self):
        test = TestFoolboxWithPytorchUsingArt()
        self.assertIsNotNone(test.test_foolbox_ProjectedGradientDescentInf())
        self.assertIsNotNone(test.test_foolbox_L1BasicIterative())
        self.assertIsNotNone(test.test_foolbox_L2BasicIterative())
        self.assertIsNotNone(test.test_foolbox_LinfBasicIterative())

    def test_ArtWithPytorchUsingArt(self):
        test = TestArtWithPytorchUsingArt()
        #self.assertIsNotNone(test.test_art_AdversarialPatch()) Adversarial Patch is not yet implemented
        self.assertIsNotNone(test.test_art_ZerothOrderOptimalization())
        self.assertIsNotNone(test.test_art_FastGradient())

    def test_ArtWithPytorchUsingFoolbox(self):
        test = TestArtWithPytorchUsingFoolbox()
        #self.assertIsNotNone(test.test_art_AdversarialPatch()) Adversarial Patch is not yet implemented
        self.assertIsNotNone(test.test_art_ZerothOrderOptimalization())
        self.assertIsNotNone(test.test_art_FastGradient())

    def test_ArtWithKerasUsingArt(self):
        test = TestArtWithKerasUsingArt()
        self.assertIsNotNone(test.test_art_ZerothOrderOptimalization())
        #self.assertIsNotNone(test.test_art_AdversarialPatch()) Adversarial Patch is not yet implemented
        self.assertIsNotNone(test.test_art_FastGradient())
