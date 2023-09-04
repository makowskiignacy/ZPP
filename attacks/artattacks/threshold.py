
from attacks.art_attack import ARTAttack
from attacks.helpers.parameters import ARTParameters

from art.attacks.evasion import ThresholdAttack


class Threshold(ARTAttack):
    '''
    Klasa otaczająca atak Threshold z biblioteki ART.
    Link do pracy - https://arxiv.org/abs/1906.06026
    '''
    def __init__(self, parameters : ARTParameters):
        '''
        Inicjalizuje atak Threshold Attack na podstawie zadanych parametrów.
        
        Możliwe parametry ataku
        -----------------------
        th ({int, None})
            Wartość progowa (opcjonalna)
        es (int)
            Czy używać CMAES (0) czy DE (1) jak strategii ewolucyjnej
            Opcja zero wymaga pakietu "cma"
        max_iter (int)
            Maksymalna liczba iteracji metody ewolucyjnej
        targeted (bool)
            Czy atak ma starać się utworzyć przykłady kontradyktoryjne, tak
            aby odpowiedzi modelu dla zadanych przykładów były
            zgodne z wartościami przekazanymi w kolumnie odpowiedzi danych.
        verbose (bool)
            Czy atak ma wypisywać informacje diagnostyczne o swoim przebiegu
        '''

        super().__init__(parameters.classifier_parameters)

        # Wartości domyślne
        self._attack_params = {
            # Wartość progowa (opcjonalna)
            "th" : None,
            # Czy używać CMAES (0) czy DE (1) jak strategii ewolucyjnej
            # Opcja zero wymaga pakietu "cma"
            "es" : 1,
            # Maksymalna liczba iteracji metody ewolucyjnej
            "max_iter" : 100,
            # Czy atak ma być celowany? To jest: czy oprócz oszukania modelu
            # ma znaleźć przykład generujący żądaną odpowiedź.
            # UWAGA: wtedy parametr procedury conduct() 'data' powinien mieć
            # w miejscu .output odpowiedzi, jaki chcemy uzyskać zmiast
            # prawdziwych odpowiedzi.
            "targeted" : False,
            # Czy wypisywać wiadomości dodatkowe z przebiegu metody ewolucyjnej?
            "verbose" : False
        }

        # Aktualizujemy, jeżeli nadpisano
        for key in self._attack_params.keys():
            if key in parameters.attack_parameters.keys():
                self._attack_params[key] = parameters.attack_parameters[key]

    def conduct(self, model, data):
        self._set_classifier(model, data)
        self._set_data(data)

        return Threshold.to_unified_format(
            ThresholdAttack(
                self._classifier,
                **self._attack_params
            ).generate(
                x = self._data.input,
                y = self._data.output
            )
        )