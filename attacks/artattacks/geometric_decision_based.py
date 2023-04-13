
from attacks.art_attack import ARTAttack
from attacks.helpers.parameters import ARTParameters

from art.attacks.evasion import GeoDA


class GeometricDecisionBased(ARTAttack):
    def __init__(self, parameters : ARTParameters):
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
        self._set_classifier(model)
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