import numpy

from attacks.art_attack import ARTAttack
from attacks.helpers.parameters import ARTParameters
from attacks.helpers.data import Data

from art.attacks.evasion import ShadowAttack


class Shadow(ARTAttack):
    def __init__(self, parameters : ARTParameters):
        super().__init__(parameters.classifier_parameters)

        # Wartości domyślne
        self._attack_params = {
            # Odchylenie standardowe rozkładu normalnego, z którego czerpie
            # się szum
            "sigma" : 0.5,
            # Liczba kroków algorytmu Stochastycznego Spadku Gradientu(ang. SGD)
            "nb_steps" : 300,
            # Wpółczynnik uczenia SGD
            "learning_rate" : 0.1,
            # Waga kary za ogołną wariancję perturbacji
            "lambda_tv" : 0.3,
            # Waga kary za zmianę średniego koloru na każdym kanale perturbacji
            "lambda_c" : 1.0,
            # Waga kary za podobieństwo kanałów koloru perturbacji
            "lambda_s" : 0.5,
            # Liczba przykładów treningowych wykorzystywanych w jednej iteracji
            # uczenia
            "batch_size" : 1,
            # Czy atak ma być celowany?
            "targeted" : False,
            # Czy pokzaywać pasek progresu?
            "verbose" : False
        }

        # Aktualizujemy, jeżeli nadpisano
        for key in self._attack_params.keys():
            if key in parameters.attack_parameters.keys():
                self._attack_params[key] = parameters.attack_parameters[key]

    def conduct(self, model, data):
        self._set_classifier(model)
        self._set_data(data)

        if self._data.input.shape[0] > 1 or self._data.output.shape[0] > 1:
            if self._data.input.shape[0] > 1 and self._data.output.shape[0] > 1:
                print("This attack only accepts a single sample as input. Using the first sample in the given data as input.")
                new_input = numpy.expand_dims(self._data.input[0], 0)
                new_output = numpy.expand_dims(numpy.asarray([self._data.output[0]]), 0)
                new_data = Data (new_input, new_output)
                self._set_data(new_data)
            else:
                raise ValueError("This attack only accepts a single sample as input. Can not automatically use the first sample, due to the shape of the data.")

        return Shadow.to_unified_format(
            ShadowAttack(
                self._classifier,
                **self._attack_params
            ).generate(
                x = self._data.input,
                y = self._data.output
            )
        )