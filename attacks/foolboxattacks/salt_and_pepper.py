from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.saltandpepper import SaltAndPepperNoiseAttack


class SaltAndPepperNoise(SaltAndPepperNoiseAttack, FoolboxAttack):
    '''
    Klasa otaczająca atak Salt and Pepper zaimplementowany w bibliotece Foolbox.
    '''
    def __init__(self, parameters):
        '''
        Inicjalizuje atak na podstawie zadanych parametróœ

        Parametry ataku:
        ----------------
        steps (int)
            Liczba kroków do wykonania.
        across_channels (bool)
            Czy szum ma być identyczny na wszystkich kanałach obrazu?
        channel_axis (Optional[int])
            Oś, względem której szum ma pozostać identyczny na wszystkich
            kanałach. Ma znaczenie tylko w przypadku ustawienia parametru
            'across_channels' na True. Jeżeli nie podano, parametr zostanie
            ustawiony automatycznie na podstawie modelu jeżeli to możliwe.
        '''
        super().__init__(**parameters.attack_specific_parameters)
        FoolboxAttack.__init__(self, parameters.generic_parameters)

    def conduct(self, model, data):

        output = super().flatten_output(data)
        super().verify_bounds(data=data)
        super().verify_epsilon()
        model_correct_format = super().reformat_model(model)

        if self.criterion_type == "targeted_misclassification":
            self.criterion = TargetedMisclassification(output)
        if self.criterion_type == "misclassification":
            self.criterion = Misclassification(output)

        result = super().run(model=model_correct_format, inputs=data.input, criterion=self.criterion)
        return result
