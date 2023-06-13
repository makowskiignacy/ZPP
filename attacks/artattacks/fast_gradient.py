"""
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""

import numpy as np
from art.attacks.evasion import FastGradientMethod

from attacks.art_attack import ARTAttack


class FastGradient(ARTAttack):
    '''
    Klasa otaczająca atak Fast Gradient Method z biblioteki ART.
    Link do pracy - https://arxiv.org/abs/1412.6572
    '''
    def __init__(self, parameters):
        '''
        Inicjalizuje atak Fast Gradient Method na podstawie zadanych parametrów.
        
        Możliwe parametry ataku
        -----------------------
        norm
            Norma w której obliczane są wielkości perturbacji zaburzonych próbek.
            Wartość ze zbioru {1,2,"inf",np.inf}
        eps –
            Wielkość pojedynczego kroku
        eps_step
            Wielkość kroku wejściowej wariacji dla obliczenia minimalnego zaburzenia.
        targeted : bool
            Czy atak ma starać się utworzyć przykłady kontradyktoryjne, tak
            aby odpowiedzi modelu dla zadanych przykładów były
            zgodne z wartościami przekazanymi w kolumnie odpowiedzi danych?
        num_random_init : int
            Liczba losowych inicjalizacji wewnątrz epsilonowej kuli naokoło
            próbki.
        batch_size : int
            Rozmiar próbki do jednorazowego treningu.
        minimal : bool
            Czy obliczać minimalne zaburzenie?
        summary_writer : bool | SummaryWriter
            Aktywuje funkcjonalność SummaryWriter modułu TensorBoard.
        '''
        # Initialization of the arguments needed for the classifier
        super().__init__(parameters.classifier_parameters)

        self._attack_params = {
            # norm – The norm of the adversarial perturbation. Possible values: “inf”, np.inf, 1 or 2.
            "norm": parameters.attack_parameters.get("norm", np.inf),
            # eps – Attack step size (input variation).
            "eps": parameters.attack_parameters.get("eps", 0.3),
            # eps_step – Step size of input variation for minimal perturbation computation.
            "eps_step": parameters.attack_parameters.get("eps_step", 0.1),
            # targeted (bool) – Indicates whether the attack is targeted (True) or untargeted (False).
            "targeted": parameters.attack_parameters.get("targeted", False),
            # num_random_init (int) – Number of random initialisations within the epsilon ball.
            "num_random_init": parameters.attack_parameters.get("num_random_init", 0),
            # batch_size (int) – Size of the batch on which adversarial samples are generated.
            "batch_size": parameters.attack_parameters.get("batch_size", 32),
            # minimal (bool) – Indicates if computing the minimal perturbation (True).
            "minimal": parameters.attack_parameters.get("minimal", False),
            # summary_writer – Activate summary writer for TensorBoard.
            "summary_writer": parameters.attack_parameters.get("summary_writer", False),
        }


    def conduct(self, model, data):
        self._set_classifier(model, data)
        self._set_data(data)

        # estimator – A trained classifier.
        return self.to_unified_format(FastGradientMethod(estimator=self._classifier, **self._attack_params).generate(
            x=self._data.input, y=self._data.output, **self._classifier_params))
