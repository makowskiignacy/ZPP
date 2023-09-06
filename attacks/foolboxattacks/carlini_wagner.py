from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.carlini_wagner import L2CarliniWagnerAttack

from attacks.helpers.data import Data
from eagerpy.astensor import astensor


class L2CarliniWagner(L2CarliniWagnerAttack, FoolboxAttack):
    '''
    Klasa otaczająca atak L2 Carliniego i Wagnera zaimplementowanego
    w bibliotece Foolbox.

    Link do pracy - https://arxiv.org/abs/1608.04644
    '''
    def __init__(self, parameters):
        '''
        Inicjalizuje atak na podstawie zadanych parametrów

        Parametry ataku:
        ----------------
        binary_search_steps (int)
            Liczba kroków wyszukiwania binarnego w celu określenia stałejk 'c'.
        steps (int)
            Liczba kroków optymalizacji wykonywanych przy każdym kroku
            wyszukiwania binarnego.
        stepsize (float)
            Wielkość kroku używana do uaktualnienia przykładów.
        confidence (float)
            Wskaźnik pewności wymagany by przykład został uznany za
            kontradyktoryjny. Ten parametr kontoluje 'odstęp' między przykładem
            a granicą decyzyjną.
        initial_const (float)
            Wartość startowa stałej 'c' na początku każdego
            wyszukiwania binarnego.
        abort_early (bool)
            Czy przerwać jak tylko odnajdzie się przykład kontradyktoryjny?
        '''
        super().__init__(**parameters.attack_specific_parameters)
        FoolboxAttack.__init__(self, parameters.generic_parameters)


    def conduct(self, model, data):
        super().verify_bounds(data=data)
        output = super().flatten_output(data)
        model_correct_format = super().reformat_model(model)

        if self.criterion_type == "targeted_misclassification":
            self.criterion = TargetedMisclassification(output)
        if self.criterion_type == "misclassification":
            self.criterion = Misclassification(output)

        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)

        return result