import unittest

from tests import input_loader_cloud, input_loader

from tests.foolbox_tests.test_brendel_bethge import TestBrendelBethge
from tests.foolbox_tests.test_basic_iterative import TestBasicIterative
from tests.foolbox_tests.test_projected_gradient_descent import TestProjectedGradientDescent
from tests.foolbox_tests.test_salt_and_pepper import TestSaltAndPepper

from tests.art_tests.test_class_hierarchy_art import TestClassHierarchyArt
from tests.art_tests.test_art_attacks import *
from tests.other_tests.test_frameworks import *

#To run the tests on a different data set, add it to input_loader.py, and then change the line below.
foolbox_model, art_model, foolbox_data, art_data, foolbox_parameters, art_parameters = input_loader_cloud.nn_input()
# foolbox_model, art_model, foolbox_data, art_data, foolbox_parameters, art_parameters = input_loader.nn_input()

class FoolboxTests(unittest.TestCase):

    def test_BrendelBethge(self):
        parameters = foolbox_parameters.get("brendel_bethge")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Brendel-Bethge attack test.")
        test = TestBrendelBethge(foolbox_model, foolbox_data, parameters)
        self.assertIsNotNone(test.test_bb_0())
        self.assertIsNotNone(test.test_bb_1())
        self.assertIsNotNone(test.test_bb_2())
        self.assertIsNotNone(test.test_bb_inf())

    def test_BasicIterative(self):
        parameters = foolbox_parameters.get("basic_iterative")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Basic Iterative attack test.")
        test = TestBasicIterative(foolbox_model, foolbox_data, parameters)
        self.assertIsNotNone(test.test_bi_1())
        self.assertIsNotNone(test.test_bi_1_a())
        self.assertIsNotNone(test.test_bi_2())
        self.assertIsNotNone(test.test_bi_2_a())
        self.assertIsNotNone(test.test_bi_inf())
        self.assertIsNotNone(test.test_bi_inf_a())

    def test_ProjectedGradientDescent(self):
        parameters = foolbox_parameters.get("projected_gradient_descent")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Projected Gradient Descent attack test.")
        test = TestProjectedGradientDescent(foolbox_model, foolbox_data, parameters)
        self.assertIsNotNone(test.test_pgd_1())
        self.assertIsNotNone(test.test_pgd_1_a())
        self.assertIsNotNone(test.test_pgd_2())
        self.assertIsNotNone(test.test_pgd_2_a())
        self.assertIsNotNone(test.test_pgd_inf())
        self.assertIsNotNone(test.test_pgd_inf_a())

    def test_SaltAndPepperNoise(self):
        parameters = foolbox_parameters.get("salt_and_pepper")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Salt and Pepper Noise attack test.")
        test = TestSaltAndPepper(foolbox_model, foolbox_data, parameters)
        self.assertIsNotNone(test.test_sap())


class ArtTests(unittest.TestCase):

    def test_ClassHierarchy(self):
        test = TestClassHierarchyArt()
        self.assertEqual(test.test_class_hierarchy(), [True, True, True])

    def test_DeepFool(self):
        parameters = art_parameters.get("deep_fool")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Deep Fool attack test.")
        test = TestDeepFool(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_FastGradient(self):
        parameters = art_parameters.get("fast_gradient")
        self.assertIsNotNone(parameters,msg="Input given for running tests does not contain parameters for the Fast Gradient attack test.")
        test = TestFastGradient(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_Joker(self):
        parameters = art_parameters.get("joker")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Joker attack test.")
        test = TestJoker(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_JacobianSaliencyMap(self):
        parameters = art_parameters.get("jacobian_saliency_map")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Jacobian Saliency Map attack test.")
        test = TestJacobianSaliencyMap(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_GeometricDecisionBased(self):
        parameters = art_parameters.get("geometric_decision_based")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Geometric Decision Based attack test.")
        test = TestGeometricDecisionBased(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_Shadow(self):
        parameters = art_parameters.get("shadow")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Shadow attack test.")
        test = TestShadow(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_Threshold(self):
        parameters = art_parameters.get("threshold")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Threshold attack test.")
        test = TestThreshold(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_SignOPT(self):
        parameters = art_parameters.get("sign_opt")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Sign-OPT attack test.")
        test = TestSignOPT(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_Square(self):
        parameters = art_parameters.get("square")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Square attack test.")
        test = TestSquare(art_model, art_data, parameters)
        self.assertIsNotNone(test.test())

    def test_ZerothOrderOptimalization(self):
        parameters = art_parameters.get("zeroth_order_optimization")
        self.assertIsNotNone(parameters, msg="Input given for running tests does not contain parameters for the Zeroth Order Optimalization attack test.")
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
