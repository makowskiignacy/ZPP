from attacks.foolbox_attack import FoolboxAttack
from foolbox.attacks import LinfPGD

# L -> inf
class ProjectedGradientDescentInf(FoolboxAttack):
    def __init__(self, **params):
        super().__init__(params)
        self._attack_params.update({'epsilons': [0.0]})
        self._attack_params.update(params)
        self._attack = LinfPGD()
        self._model = None

    def conduct(self, model, data):
        self._model = super().reformat_model(model)
        return self.to_unified_format(self._attack(self._model, data.input, data.output, **self._attack_params))
