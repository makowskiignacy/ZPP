import traceback

from foolbox.attacks.brendel_bethge import L0BrendelBethgeAttack, L1BrendelBethgeAttack, L2BrendelBethgeAttack, LinfinityBrendelBethgeAttack

from attacks.foolbox_attack import FoolboxAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from abc import ABC


class GenericBrendelBethge(FoolboxAttack, ABC):
    """
    Klasa generyczna dla ataku typu Brendel Bethge z biblioteki Foolbox.
    Jest to 'silny' atak typu gradientowego, który stara się podążać wzdłuż
    granicy między obrazmi prawidłowymi a kontradyktoryjnymi i znaleźć przykład
    najmniej odległy od oryginalnego obrazu.

    Link do pracy - https://arxiv.org/abs/1907.01003
    """
    def __init__(self, parent, parameters):
        '''
        Inicjalizuje obiekt na podstawie wybranego rodzaju ataku Brendel Bethge

        Parametry:
        ----------
        parent (foolbox.attacks.brendel_bethge)
            Rodzaj ataku z rodziny Brendel Bethge do zainicjalizowania
        parameters
            Parametry odpowiednie dla wybranego ataku

        Parametry ataku:
        ----------------
        init_attack (Optional[foolbox.attacks.base.MinimizationAttack]) –
        overshoot (float) –
        steps (int) –
        lr (float) –
        lr_decay (float) –
        lr_num_decay (int) –
        momentum (float) –
        tensorboard (Union[typing_extensions.Literal[False], None, str]) –

        binary_search_steps (int) –
        '''
        self.Parent = parent
        self.Parent.__init__(self, **parameters.attack_specific_parameters)
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

        result = self.Parent.run(self, model=model_correct_format, inputs=data.input, criterion=self.criterion)
        return result



class L0BrendelBethge(GenericBrendelBethge, L0BrendelBethgeAttack):
    '''
    Atak Brendel Bethge mierzący odległość w normie L0.
    '''
    def __init__(self, parameters):
        GenericBrendelBethge.__init__(self, L0BrendelBethgeAttack, parameters)


class L1BrendelBethge(GenericBrendelBethge, L1BrendelBethgeAttack):
    '''
    Atak Brendel Bethge mierzący odległość w normie L1.
    '''
    def __init__(self, parameters):
        GenericBrendelBethge.__init__(self, L1BrendelBethgeAttack, parameters)


class L2BrendelBethge(GenericBrendelBethge, L2BrendelBethgeAttack):
    '''
    Atak Brendel Bethge mierzący odległość w normie L2.
    '''
    def __init__(self, parameters):
        GenericBrendelBethge.__init__(self, L2BrendelBethgeAttack, parameters)


class LinfBrendelBethge(GenericBrendelBethge, LinfinityBrendelBethgeAttack):
    '''
    Atak Brendel Bethge mierzący odległość w normie L-nieskończoność.
    '''
    def __init__(self, parameters):
        GenericBrendelBethge.__init__(self, LinfinityBrendelBethgeAttack, parameters)
