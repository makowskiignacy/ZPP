from copyreg import constructor
from attacks.attack import Attack

# Ataki pochodzące z ARTa
from attacks.artattacks.deep_fool import DeepFool
from attacks.artattacks.fast_gradient import FastGradient
from attacks.artattacks.joker import Joker

from attacks.artattacks.zeroth_order_optimization_bb_attack\
    import ZerothOrderOptimalization as ZOOAttack

from attacks.artattacks.adversarial_patch\
    import AdversarialPatch

from attacks.artattacks.geometric_decision_based import GeometricDecisionBased
from attacks.artattacks.jacobian_saliency_map import JacobianSaliencyMap
from attacks.artattacks.square import Square
from attacks.artattacks.sign_opt import SignOPT
from attacks.artattacks.threshold import Threshold

# Ataki pochodzące z FoolBoxa
from attacks.foolboxattacks.additive_noise import L2AdditiveGaussianNoise, L2ClippingAwareAdditiveGaussianNoise, L2RepeatedAdditiveGaussianNoise, L2ClippingAwareRepeatedAdditiveGaussianNoise, LinfAdditiveUniformNoise, LinfRepeatedAdditiveUniformNoise
from attacks.foolboxattacks.additive_noise import L2AdditiveUniformNoise, L2ClippingAwareAdditiveUniformNoise, L2RepeatedAdditiveUniformNoise, L2ClippingAwareRepeatedAdditiveUniformNoise
from attacks.foolboxattacks.brendel_bethge import L0BrendelBethge, L1BrendelBethge, L2BrendelBethge, LinfBrendelBethge
from attacks.foolboxattacks.basic_iterative import L1BasicIterative, L2BasicIterative, LinfBasicIterative
from attacks.foolboxattacks.basic_iterative import L1AdamBasicIterative, L2AdamBasicIterative, LinfAdamBasicIterative
from attacks.foolboxattacks.carlini_wagner import L2CarliniWagner
from attacks.foolboxattacks.newton_fool import NewtonFool
from attacks.foolboxattacks.projected_gradient_descent import L1ProjectedGradientDescent, L2ProjectedGradientDescent, LinfProjectedGradientDescent
from attacks.foolboxattacks.projected_gradient_descent import L1AdamProjectedGradientDescent, L2AdamProjectedGradientDescent, LinfAdamProjectedGradientDescent
from attacks.foolboxattacks.salt_and_pepper import SaltAndPepperNoise
from attacks.helpers.parameters import ARTParameters, FoolboxParameters


class AttackNameNotRecognized(Exception):
    def __init__(self, name):
        super().__init__(f"Attack name \"{name}\" not recognized.")

class AttackListEntry():
    def __init__(self, constructor, default_params):
        self.constructor = constructor,
        self.default_params = default_params

class AttackManager:
    
    classifier_parameters_default = {"clip_values": (-2., 30000.) }
    attack_parameters_default = {}
    art_parameters_default = ARTParameters(classifier_parameters_default, attack_parameters_default)

    default_generic_parameters_an = {"epsilon_rate": 0.01}
    default_attack_specific_parameters_an = {}
    default_foolbox_parameters_an = FoolboxParameters(default_attack_specific_parameters_an, default_generic_parameters_an)
    
    default_generic_parameters_bb = {"epsilon_rate": 0.01}
    default_attack_specific_parameters_bb = {"lr": 10, 'steps': 100}
    default_foolbox_parameters_bb = FoolboxParameters(default_attack_specific_parameters_bb, default_generic_parameters_bb)

    default_generic_parameters_bi = {"epsilon_rate": 0.05}
    default_attack_specific_parameters_bi = {"steps": 100, "random_start": True}
    default_foolbox_parameters_bi = FoolboxParameters(default_attack_specific_parameters_bi, default_generic_parameters_bi)

    default_generic_parameters_cw = {"epsilon_rate": 0.01}
    default_attack_specific_parameters_cw = {"steps": 100}
    default_foolbox_parameters_cw = FoolboxParameters(default_attack_specific_parameters_cw, default_generic_parameters_cw)

    default_generic_parameters_nf = {"epsilon_rate": 0.01}
    default_attack_specific_parameters_nf = {"steps": 100, "stepsize": 100}
    default_foolbox_parameters_nf = FoolboxParameters(default_attack_specific_parameters_nf, default_generic_parameters_nf)

    default_generic_parameters_pgd = {"epsilon_rate": 0.01}
    default_attack_specific_parameters_pgd = {"steps": 100, "random_start": True}
    default_foolbox_parameters_pgd = FoolboxParameters(default_attack_specific_parameters_pgd, default_generic_parameters_pgd)

    default_generic_parameters_sap = {"epsilon_rate": 0.01}
    default_attack_specific_parameters_sap = {"steps": 100, "across_channels": True}
    default_foolbox_parameters_sap = FoolboxParameters(default_attack_specific_parameters_sap, default_generic_parameters_sap)


    # Takie użycie zapewnia niezmienniczość listy w trakcie pracy programu
    @staticmethod
    def get_possible_attacks():
        return {
            # możliwe, że elementami będą pary (klasa, procedura sprawdzająca)

            "Adversarial Patch"         : AttackListEntry(AdversarialPatch, AttackManager.art_parameters_default),
            "DeepFool"                  : AttackListEntry(DeepFool, AttackManager.art_parameters_default),
            "FastGradient"              : AttackListEntry(FastGradient, AttackManager.art_parameters_default),
            "Geometric Decision Based"  : AttackListEntry(GeometricDecisionBased, AttackManager.art_parameters_default),
            "Jacobian Saliency Map"     : AttackListEntry(JacobianSaliencyMap, AttackManager.art_parameters_default),
            "Joker"                     : AttackListEntry(Joker, AttackManager.art_parameters_default),
            "L2 Additive Gaussian Noise": AttackListEntry(L2AdditiveGaussianNoise, AttackManager.default_foolbox_parameters_an),
            "L2 Additive Uniform Noise" : AttackListEntry(L2AdditiveUniformNoise, AttackManager.default_foolbox_parameters_an),
            "L2 Clipping Aware Additive Gaussian Noise"
                                        : AttackListEntry(L2ClippingAwareAdditiveGaussianNoise, AttackManager.default_foolbox_parameters_an),
            "L2 Clipping Aware Additive Uniform Noise"
                                        : AttackListEntry(L2ClippingAwareAdditiveUniformNoise, AttackManager.default_foolbox_parameters_an),
            "L2 Repeated Additive Gaussian Noise"
                                        : AttackListEntry(L2RepeatedAdditiveGaussianNoise, AttackManager.default_foolbox_parameters_an),
            "L2 Repeated Additive Uniform Noise"
                                        : AttackListEntry(L2RepeatedAdditiveUniformNoise, AttackManager.default_foolbox_parameters_an),
            "L2 Clipping Aware Repeated Additive Gaussian Noise"
                                        : AttackListEntry(L2ClippingAwareRepeatedAdditiveGaussianNoise, AttackManager.default_foolbox_parameters_an),
            "L2 Clipping Aware Repeated Additive Uniform Noise"
                                        : AttackListEntry(L2ClippingAwareRepeatedAdditiveUniformNoise, AttackManager.default_foolbox_parameters_an),
            "Linf Additive Uniform Noise"
                                        : AttackListEntry(LinfAdditiveUniformNoise, AttackManager.default_foolbox_parameters_an),
            "Linf Repeated Additive Uniform Noise"
                                        : AttackListEntry(LinfRepeatedAdditiveUniformNoise, AttackManager.default_foolbox_parameters_an),
            "L0 Brendel Bethge"         : AttackListEntry(L0BrendelBethge, AttackManager.default_foolbox_parameters_bb),
            "L1 Brendel Bethge"         : AttackListEntry(L1BrendelBethge, AttackManager.default_foolbox_parameters_bb),
            "L2 Brendel Bethge"         : AttackListEntry(L2BrendelBethge, AttackManager.default_foolbox_parameters_bb),
            "Linf Brendel Bethge"       : AttackListEntry(LinfBrendelBethge, AttackManager.default_foolbox_parameters_bb),
            "L1 Basic Iterative"        : AttackListEntry(L1BasicIterative, AttackManager.default_foolbox_parameters_bi),
            "L1 Adam Basic Iterative"   : AttackListEntry(L1AdamBasicIterative, AttackManager.default_foolbox_parameters_bi),
            "L2 Adam Basic Iterative"   : AttackListEntry(L2AdamBasicIterative, AttackManager.default_foolbox_parameters_bi),
            "Linf Adam Basic Iterative" : AttackListEntry(LinfAdamBasicIterative, AttackManager.default_foolbox_parameters_bi),
            "Linf Basic Iterative"      : AttackListEntry(LinfBasicIterative, AttackManager.default_foolbox_parameters_bi),
            "L2 Basic Iterative"        : AttackListEntry(L2BasicIterative, AttackManager.default_foolbox_parameters_bi),
            "L2 Carlini-Wagner"         : AttackListEntry(L2CarliniWagner, AttackManager.default_foolbox_parameters_cw),
            "L1 Adam Projected Gradient Descent"
                                       : AttackListEntry(L1AdamProjectedGradientDescent, AttackManager.default_foolbox_parameters_pgd),
            "L1 Projected Gradient Descent"
                                       : AttackListEntry(L1ProjectedGradientDescent, AttackManager.default_foolbox_parameters_pgd),
            "L2 Adam Projected Gradient Descent"
                                       : AttackListEntry(L2AdamProjectedGradientDescent, AttackManager.default_foolbox_parameters_pgd),
            "L2 Projected Gradient Descent"
                                       : AttackListEntry(L2ProjectedGradientDescent, AttackManager.default_foolbox_parameters_pgd),
            "Linf Adam Projected Gradient Descent"
                                       : AttackListEntry(LinfAdamProjectedGradientDescent, AttackManager.default_foolbox_parameters_pgd),
            "Linf Projected Gradient Descent"
                                       : AttackListEntry(LinfProjectedGradientDescent, AttackManager.default_foolbox_parameters_pgd),
            "Newton Fool"              : AttackListEntry(NewtonFool, AttackManager.default_foolbox_parameters_nf),
            "Salt And Pepper"          : AttackListEntry(SaltAndPepperNoise, AttackManager.default_foolbox_parameters_sap),
            "Sign-OPT"                 : AttackListEntry(SignOPT, AttackManager.art_parameters_default),
            "Square"                   : AttackListEntry(Square, AttackManager.art_parameters_default),
            "Threshold"                : AttackListEntry(Threshold, AttackManager.art_parameters_default),
            "Zeroth Order Optimalization"
                                       : AttackListEntry(ZOOAttack, AttackManager.art_parameters_default)
        }
    
    @staticmethod
    def __get_error_codes() -> dict[int, str]:
        return {
            0xBADC0DE : "Example of error code"
        }

    # Tworzy i TODO sprawdza poprawność parametrów podanych ataków
    @staticmethod
    def create_attacks(attack_names_params_list: list[tuple[str, dict[str, ]]]) -> list:

        for name, _ in attack_names_params_list:
            if name not in AttackManager.get_possible_attacks():
                raise AttackNameNotRecognized(name)

        attacks = []
        for name, params in attack_names_params_list:
            if name in AttackManager.get_possible_attacks():
                # TODO sprawdzenie parametrów ataku
                entry = AttackManager.get_possible_attacks()[name]
                final_params = entry.default_params
                if params is not None:
                    final_params.update(params)
                    
                attacks.append(
                    # Wywołanie konstruktora ataku pod nazwą "name"
                    entry.constructor[0](params)
                )
        return attacks
    
    def __init__(self):
        # Inicjalizacja odpowiednich atrybutów, tak aby możliwa była
        # optymalizacja wywołania conduct_attacks
        pass
    
    # Przeprowadza zadane ataki na podanym modelu i danych wejściowych
    # NOTE powinno być to zrobione w możliwe zoptymalizowany sposób
    def conduct_attacks(self, attacks: list[Attack], model, data):
        # TODO Zrobić to w zoptymalizowany sposób
        results = []
        # zakładam kolejność wywoływania pozostaje niezmieniona
        # inaczej trzeba dodać id ataku do pozycji w wynikach
        for attack in attacks:
            results.append(attack.conduct(model, data))
        
        return results
    
    