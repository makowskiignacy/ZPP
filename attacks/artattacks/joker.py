# Universal substitute class for ART to perform attacks not yet implemented.
# Searches global variables for the class name corresponding to the joker parameter.

from attacks.art_attack import ARTAttack

# Importing classes
from art.attacks.evasion import *

class Joker(ARTAttack):
    def __init__(self, joker, parameters):
        self.joker=joker
        super().__init__(parameters.classifier_parameters)
        self._attack_params = parameters.attack_parameters

    def conduct(self, model, data):
        self._set_classifier(model)
        self._set_data(data)

        return self.to_unified_format(globals()[self.joker](estimator=self._classifier, **self._attack_params).generate(
            x=self._data.input, y=self._data.output, **self._classifier_params))