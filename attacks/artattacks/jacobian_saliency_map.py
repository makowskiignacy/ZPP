
from attacks.art_attack import ARTAttack
from attacks.helpers.parameters import ARTParameters

from art.attacks.evasion import SaliencyMapMethod


class JacobianSaliencyMap(ARTAttack):
    def __init__(self,parameters : ARTParameters) :
        super().__init__(parameters.classifier_parameters)

        # Wartości domyślne
        self._attack_params = {
            # Wartość perturbacji każdej zmodyfikowanej cechy na jeden krok
            "theta" : 0.1,
            # Maksymalny ułamek cech jakie będą poddane perturbacji
            "gamma" : 1.0,
            # Wielkość gruby próbek na jakiej będą generowane przykłady kontradyktoryjne
            "batch_size" : 1,
            # Czy pokazywać pasek progresu
            "verbose" : False
        }
        # Aktualizujemy, jeżeli nadpisano
        self._attack_params.update(parameters.attack_parameters)

    def conduct(self, model, data):
        self._set_classifier(model)
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