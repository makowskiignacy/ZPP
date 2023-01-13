from attacks.artattacks.zeroth_order_optimization_bb_attack\
    import ZeorthOrderOptimalization as ZOOAttack

from attacks.artattacks.adversarial_patch\
    import AdversarialPatch 

from attacks.attack import Attack


class AttackManager():
    # Takie użycie zapewnia niezmienniczość listy w trakcie pracy programu
    @staticmethod
    def get_possible_attacks():
        return {
            # możliwe, że elementami będą pary (klasa, procedura sprawdzająca)
            "Zeorth Order Optimalization" : ZOOAttack,
            "AdversarialPatch" : AdversarialPatch
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
        