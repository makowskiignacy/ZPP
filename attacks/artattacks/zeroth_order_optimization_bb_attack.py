from attacks.art_attack import ARTAttack
from art.attacks.evasion import ZooAttack as orig_art_zoo_attack


class ZeorthOrderOptimalization(ARTAttack):
    def __init__(self, **params):
        # inicjalizacja argumentów potrzebnych dla klasyfikatora, będą one wspólne dla wszystkich ataków
        super().__init__(**params)
        # Definiujemy podstawowe parametry ataku
        self._attack_params.update({
            # Pewność przykładów kontradyktoryjnych
            'confidence': 0.0,
            # Czy atak ma sie skupić na konkrentej klasie
            'targeted': False,
            # mniej => lepsze wyniki ale dłużej
            'learning_rate': 0.01,
            # To można zwiększyć
            'max_iter': 10,
            # To również
            'binary_search_steps': 1,
            # To nie ma znaczenia, gdy powyższe jest 'duże'
            'initial_const': 0.001,
            # Czy w przypadku 'utknięcia' przerwać wcześniej
            'abort_early': True,

            'use_resize': True,

            'use_importance': True,
            # To można chyba podkręcić
            'nb_parallel': 128,
            # Wielkość grup próbek (warto zwrócić uwagę, że wykonywanych jest
            # nb_parallel na raz, stąd nie powinno być bardzo duże)
            'batch_size': 1,
            # Zmienna używana do numerycznego przybliżenia pochodnej
            'variable_h': 0.0001,
            # Czy pokazywać pasek progresu
            'verbose': False
        })
        # Jeśli podano inne to podmieniamy, ale nie uwzględniamy
        # parametrów spoza listy!
        for key in self._attack_params.keys():
            if key in params.keys():
                self._attack_params[key] = params[key]

    def conduct(self, model, data):
        # Ustawiamy atakowany model
        self._set_classifier(model)
        # Ustawiamy atakowany zbiór
        self._set_data(data)

        # Usuwamy niepotrzebne parametry - konieczne, żeby nie rzucało błędów
        for i in ['mask', 'reset_patch', 'input_shape', 'loss', 'nb_classes', 'optimizer', 'clip_values']:
            del self._attack_params[i]

        return ZeorthOrderOptimalization.to_unified_format(
            # Ważne, aby dodać model do listy parametrów podczas tworzenia
            # obiektu z biblioteki ART
            orig_art_zoo_attack(
                classifier=self._classifier, **self._attack_params
            ).generate(
                self._data.input,
                self._data.output
            )
        )