
from attacks.art_attack import ARTAttack
from attacks.helpers.parameters import ARTParameters

from art.attacks.evasion import SignOPTAttack


class SignOPT(ARTAttack):
    def __init__(self, parameters : ARTParameters):
        super().__init__(parameters.classifier_parameters)

        # Wartości domyślne
        self._attack_params = {
            # Parametr wygładzający (b. mały)
            "epsilon" : 0.001,
            # Liczba prób wyliczenia dobrego punktu startowego
            "num_trial" : 100,
            # Maksymalna liczba iteracji. Dobrze zwiększyć dla ataku celowanego
            "max_iter" : 1000,
            # Limit zapytań do modelu. Dobrze zwiększyć dla ataku celowanego
            "query_limit" : 20000,
            # Liczb losowych kierunków używanych do estymacji gradientu
            "k" : 200,
            # Długość kroku wyszukiwania liniowego
            "alpha" : 0.2,
            # Tolerancja wyszukiwania liniowego
            "beta" : 0.001,
            # Rozmiar partii używany przez estymator podczas wnioskowania
            "batch_size" : 64,
            # Czy atak ma skupiać się na pojedynczej klasie
            "targeted" : False,
            # Czy pokazywać bardziej detaliczne informacje?
            "verbose" : False,
            # Czy ocenić wydajność na 100 losowo wygenerowanych przykładach?
            "eval_perform" : False
        }

        # Aktualizujemy, jeżeli nadpisano
        for key in self._attack_params.keys():
            if key in parameters.attack_parameters.keys():
                self._attack_params[key] = parameters.attack_parameters[key]

    def conduct(self, model, data):
        self._set_classifier(model, data)
        self._set_data(data)

        return SignOPT.to_unified_format(
            SignOPTAttack(
                self._classifier,
                **self._attack_params
            ).generate(
                x = self._data.input,
                y = self._data.output
            )
        )