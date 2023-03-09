# Universal substitute class for ART to perform attacks not yet implemented.
# Searches global variables for the class name corresponding to the joker parameter.

from attacks.art_attack import ARTAttack

# Importing classes
from art.attacks.evasion import *

class Joker(ARTAttack):
    def __init__(self, **params):
        if 'joker' not in params:
            raise Exception("The joker parameter is mandatory, set it to the name of the desired class from ART")

        # Initialization of the arguments needed for the classifier
        super().__init__(**params)

        for key in self._classifier_params:
            if key in params:
                params.pop(key)

        self.joker = params.pop('joker')
        self._attack_params = params

    def conduct(self, model, data):
        self._set_classifier(model)
        self._set_data(data)

        return self.to_unified_format(globals()[self.joker](estimator=self._classifier, **self._attack_params).generate(
            x=self._data.input, y=self._data.output, **self._classifier_params))