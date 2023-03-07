"""
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""

import numpy as np
from art.attacks.evasion import FastGradientMethod

from attacks.art_attack import ARTAttack


class FastGradient(ARTAttack):
    def __init__(self, **params):
        # Initialization of the arguments needed for the classifier
        super().__init__(**params)

        self._attack_params = {
            # norm – The norm of the adversarial perturbation. Possible values: “inf”, np.inf, 1 or 2.
            "norm": np.inf,
            # eps – Attack step size (input variation).
            "eps": 0.3,
            # eps_step – Step size of input variation for minimal perturbation computation.
            "eps_step": 0.1,
            # targeted (bool) – Indicates whether the attack is targeted (True) or untargeted (False).
            "targeted": False,
            # num_random_init (int) – Number of random initialisations within the epsilon ball.
            "num_random_init": 0,
            # batch_size (int) – Size of the batch on which adversarial samples are generated.
            "batch_size": 32,
            # minimal (bool) – Indicates if computing the minimal perturbation (True).
            "minimal": False,
            # summary_writer – Activate summary writer for TensorBoard.
            "summary_writer": False
        }

        # Assigning only relevant arguments
        for key in self._attack_params.keys():
            if key in params.keys():
                self._attack_params[key] = params[key]

    def conduct(self, model, data):
        self._set_classifier(model)
        self._set_data(data)

        # estimator – A trained classifier.
        return self.to_unified_format(FastGradientMethod(estimator=self._classifier, **self._attack_params).generate(
            x=self._data.input, y=self._data.output, **self._attack_params))