
from attacks.art_attack import ARTAttack
from attacks.helpers.parameters import ARTParameters

from art.attacks.evasion import SquareAttack


class Square(ARTAttack):
    '''
    Klasa otaczająca atak Square z biblioteki ART.
    Link do pracy - https://arxiv.org/abs/1912.00049
    '''
    def __init__(self,parameters : ARTParameters) :
        super().__init__(parameters.classifier_parameters)
        '''
        Inicjalizuje atak Square Attack na podstawie zadanych parametrów.
        
        Możliwe parametry ataku
        -----------------------
        norm ({1,2,"inf"})
            Norma L_, możliwe wartości to 1,2,"inf"
        adv_criterion (Callable[[ndarray,ndarray], bool])
            Funkcja jakiej atak ma użyć aby określić 'kontradyktoryjność'
            Domyślnie 'None'
        loss (Callable[[ndarray,ndarray], numpy.ndarray])
            Funkcja, którą atak ma optymalizować
        max_iter (int)
            Maksymalna liczba iteracji
        eps (float)
            Maksymalna dozwolona perturbacja
        p_init (float)
            Wstępna część elementów
        nb_restarts (int)
            Liczba ponownych uruchomień
        batch_size (int)
            Liczba przykładów treningowych wykorzystywanych w jednej iteracji
            uczenia
        verbose (bool)
            Czy atak ma wypisywać informacje diagnostyczne o swoim przebiegu
        '''
        # Wartości domyślne
        self._attack_params = {
            # Norma L_, możliwe wartości to 1,2,"inf"
            "norm" : 2,
            # Funkcja jakiej atak ma użyć aby określić 'kontradyktoryjność'
            "adv_criterion" : None,
            # Funkcja, którą atak ma optymalizować
            "loss" : None,
            # Maksymalna liczba iteracji
            "max_iter" : 100,
            # Maksymalna dozwolona perturbacja
            "eps" : 0.3,
            # Wstępna część elementów
            "p_init": 0.8,
            # Liczba ponownych uruchomień
            "nb_restarts" : 1,
            # Liczba przykładów treningowych wykorzystywanych w jednej iteracji
            # uczenia
            "batch_size" : 128,
            # Czy pokazywać pasek progresu
            "verbose" : False
        }
        
        # Aktualizujemy, jeżeli nadpisano
        for key in self._attack_params.keys():
            if key in parameters.attack_parameters.keys():
                self._attack_params[key] = parameters.attack_parameters[key]

    # Ten atak ma tylko postać 'celowaną' to znaczy, że podane wartości
    # data.output są traktowane jako docelowe odpowiedzi modelu
    def conduct(self, model, data):
        self._set_classifier(model, data)
        self._set_data(data)

        return Square.to_unified_format(
            SquareAttack(
                self._classifier,
                **self._attack_params
            ).generate(
                x = self._data.input,
                y = self._data.output
            )
        )