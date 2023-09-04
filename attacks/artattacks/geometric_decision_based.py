
from attacks.art_attack import ARTAttack
from attacks.helpers.parameters import ARTParameters

from art.attacks.evasion import GeoDA


class GeometricDecisionBased(ARTAttack):
    '''
    Klasa otaczająca atak Geometric Decision Based z biblioteki ART.
    Link do pracy - https://arxiv.org/abs/2003.06468
    '''
    def __init__(self, parameters : ARTParameters):
        '''
        Inicjalizuje atak Geometric Decision Based na podstawie zadanych parametrów.
        
        Możliwe parametry ataku
        -----------------------
        # Używana norma o możliwych wartościach: "inf",1,2
        norm
            Norma w której obliczane są wielkości perturbacji zaburzonych próbek.
            Wartość ze zbioru {1,2,"inf",np.inf}
        sub_dim
            Wymiarowość przestrzeni cześtostliwości 2D w dyskretnej
            transformacji cosinusów
        max_iter : int
            Maksymalna liczba iteracji algorytmu.
        bin_search_tol : float
            Maksymalna pozostała norma L2 perturbacji określająca
            zbieżność wyszukiwania binarnego.
        lambda_param : float
            Parametr lambda z równiania (19). Lambda = 0 odpowiada
            pojdeynczej iteracji, podczas gdy lambda = 1 odpowiada
            jednorodnemu rozkładowi iteracji na krok.
        sigma : 0.0002,
            Wariancja perturbacji Gaussowskiej (rozkład normalny)
        batch_size : int
            Liczba przykładów treningowych wykorzystywanych w jednej iteracji
            uczenia
        verbose (bool)
            Czy atak ma wypisywać informacje diagnostyczne o swoim przebiegu
        '''
        super().__init__(parameters.classifier_parameters)

        # Wartości domyślne
        self._attack_params = {
            # Używana norma o możliwych wartościach: "inf",1,2
            "norm" : 2,
            # Wymiarowość przestrzeni cześtostliwości 2D w
            # dyskretnej transformacji cosinusów
            "sub_dim" : 10,
            # Maksymalna liczba iteracji.
            "max_iter" : 4000,
            # Maksymalna pozostała norma L2 perturbacji określająca
            # zbieżność wyszukiwania binarnego
            "bin_search_tol" : 0.1,
            # Parametr lambda z równiania (19). Lambda = 0 odpowiada
            # pojdeynczej iteracji, podczas gdy lambda = 1 odpowiada
            # jednorodnemu rozkładowi iteracji na krok.
            "lambda_param" : 0.6,
            # Wariancja perturbacji Gaussowskiej (rozkład normalny?)
            "sigma" : 0.0002,
            # Ten parametr jest teoretycznie wspierany, natomiast nie jest
            # wspierany przez API, stąd musimy ten fragm. wykomentować
            # Czy atak ma być celowany?
            # "targeted" : False,
            # Rozmiar partii używany przez estymator podczas wnioskowania
            "batch_size" : 64,
            # Czy pokazywać pasek progresu?
            "verbose" : False
        }

        # Aktualizujemy, jeżeli nadpisano
        for key in self._attack_params.keys():
            if key in parameters.attack_parameters.keys():
                self._attack_params[key] = parameters.attack_parameters[key]

    def conduct(self, model, data):
        self._set_classifier(model, data)
        self._set_data(data)

        return GeometricDecisionBased.to_unified_format(
            GeoDA(
                self._classifier,
                **self._attack_params
            ).generate(
                x = self._data.input,
                y = self._data.output
            )
        )