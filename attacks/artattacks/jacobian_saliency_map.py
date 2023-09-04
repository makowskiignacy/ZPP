
from attacks.art_attack import ARTAttack
from attacks.helpers.parameters import ARTParameters

from art.attacks.evasion import SaliencyMapMethod


class JacobianSaliencyMap(ARTAttack):
    '''
    Klasa otaczająca atak Jacobian Saliency Map z biblioteki ART.
    Link do pracy - https://arxiv.org/abs/1511.07528
    '''
    def __init__(self,parameters : ARTParameters) :
        '''
        Inicjalizuje atak Jacobian Saliency Map na podstawie zadanych parametrów.

        Możliwe parametry ataku
        -----------------------
        theta (float)
            Wartość perturbacji każdej zmodyfikowanej cechy na jeden krok
        gamma (float)
            Maksymalny ułamek cech jakie będą poddane perturbacji.
            Oczekiwana wartość między 0 a 1
        batch_size (int)
            Liczba przykładów treningowych wykorzystywanych w jednej iteracji
            uczenia
        verbose (bool)
            Czy atak ma wypisywać informacje diagnostyczne o swoim przebiegu
        '''
        super().__init__(parameters.classifier_parameters)

        # Wartości domyślne
        self._attack_params = {
            # Wartość perturbacji każdej zmodyfikowanej cechy na jeden krok
            "theta" : 0.1,
            # Maksymalny ułamek cech jakie będą poddane perturbacji
            "gamma" : 1.0,
            # Liczba przykładów treningowych wykorzystywanych w jednej iteracji
            # uczenia
            "batch_size" : 1,
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

        return JacobianSaliencyMap.to_unified_format(
            SaliencyMapMethod(
                self._classifier,
                **self._attack_params
            ).generate(
                x = self._data.input,
                y = self._data.output
            )
        )