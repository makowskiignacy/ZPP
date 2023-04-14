from attacks.art_attack import ARTAttack
from art.attacks.evasion import ZooAttack as orig_art_zoo_attack



class ZeroethOrderOptimalization(ARTAttack):
    def __init__(self, parameters):
        # inicjalizacja argumentów potrzebnych dla klasyfikatora, będą one wspólne dla wszystkich ataków
        super().__init__(parameters.classifier_parameters)
        # Definiujemy podstawowe parametry ataku
        self._attack_params = {
            # Pewność przykładów kontradyktoryjnych
            "confidence": parameters.attack_parameters.get("confidence", 0.0),
            # Czy atak ma sie skupić na konkrentej klasie
            "targeted": parameters.attack_parameters.get("targeted", False),
            # mniej => lepsze wyniki ale dłużej
            "learning_rate": parameters.attack_parameters.get("learning_rate", 0.01),
            # To można zwiększyć
            "max_iter": parameters.attack_parameters.get("max_iter", 10),
            # To również
            "binary_search_steps": parameters.attack_parameters.get("binary_search_steps", 1),
            # To nie ma znaczenia, gdy powyższe jest 'duże'
            "initial_const": parameters.attack_parameters.get("initial_const", 0.001),
            # Czy w przypadku 'utknięcia' przerwać wcześniej
            "abort_early": parameters.attack_parameters.get("abort_early", True),
            "use_resize": parameters.attack_parameters.get("use_resize", True),
            "use_importance": parameters.attack_parameters.get("use_importance", True),
            # To można chyba podkręcić
            "nb_parallel": parameters.attack_parameters.get("nb_parallel", 128),
            # Wielkość grup próbek (warto zwrócić uwagę, że wykonywanych jest nb_parallel na raz, stąd nie powinno być bardzo duże)
            "batch_size": parameters.attack_parameters.get("batch_size", 1),
            # Zmienna używana do numerycznego przybliżenia pochodnej
            "variable_h": parameters.attack_parameters.get("variable_h", 0.0001),
            # Czy pokazywać pasek progresu
            "verbose": parameters.attack_parameters.get("verbose", False),
        }



    def conduct(self, model, data):
        # Ustawiamy atakowany model
        self._set_classifier(model)
        # Ustawiamy atakowany zbiór
        self._set_data(data)

        # Narazie odkomentowane, do usunięcia przy przyjęciu poprawek z art_attack.py
        #Usuwamy niepotrzebne parametry - konieczne, żeby nie rzucało błędów
        # for i in ['mask', 'reset_patch', 'input_shape', 'loss', 'nb_classes', 'optimizer', 'clip_values']:
        #     del self._attack_params[i]

        return ZeroethOrderOptimalization.to_unified_format(
            # Ważne, aby dodać model do listy parametrów podczas tworzenia
            # obiektu z biblioteki ART
            orig_art_zoo_attack(
                classifier=self._classifier, **self._attack_params
            ).generate(
                x = self._data.input,
                y = self._data.output
            )
        )