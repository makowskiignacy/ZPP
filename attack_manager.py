from attacks.attack import Attack

# Ataki pochodzące z ARTa
from attacks.artattacks.deep_fool import DeepFool
from attacks.artattacks.fast_gradient import FastGradient
from attacks.artattacks.joker import Joker

from attacks.artattacks.zeroth_order_optimization_bb_attack\
    import ZeorthOrderOptimalization as ZOOAttack

from attacks.artattacks.adversarial_patch\
    import AdversarialPatch

# Ataki pochodzące z FoolBoxa
from attacks.foolboxattacks.brendel_bethge import BrendelBethge 
from attacks.foolboxattacks.L1_basic_iterative import L1BasicIterative
from attacks.foolboxattacks.projected_gradient_descent\
        import ProjectedGradientDescentInf
from attacks.foolboxattacks.salt_and_pepper import SaltAndPepperNoise





class AttackManager():
    # Takie użycie zapewnia niezmienniczość listy w trakcie pracy programu
    @staticmethod
    def get_possible_attacks():
        return {
            # możliwe, że elementami będą pary (klasa, procedura sprawdzająca)
            "Zeorth Order Optimalization" : ZOOAttack,
            "Adversarial Patch" : AdversarialPatch,
            "DeepFool" : DeepFool,
            "FastGradient" : FastGradient,
            "Joker" : Joker,
            "Brendel Bethge" : BrendelBethge,
            "L1 Basic Iterative": L1BasicIterative,
            "Projected Gradient Descent Inf" : ProjectedGradientDescentInf,
            "Salt And Pepper" : SaltAndPepperNoise
        }
    
    @staticmethod
    def __get_error_codes() -> dict[int, str]:
        return {
            0xBADC0DE : "Example of error code"
        }

    # Tworzy i TODO sprawdza poprawność parametrów podanych ataków
    @staticmethod
    def create_attacks(attack_names_params_list: list) -> list:
        attacks = []
        for name, params in attack_names_params_list:
            if name in AttackManager.get_possible_attacks():
                # TODO sprawdzenie parametrów ataku
                attacks.append(
                    # Wywołanie konstruktora ataku pod nazwą "name"
                    AttackManager.get_possible_attacks()[name](params)
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
            results.append(attack.conduct(model, data))\
        
        return results
        