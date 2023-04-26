from attacks.foolboxattacks.salt_and_pepper import SaltAndPepperNoise

from attacks.attack import run_attack


class TestSaltAndPepper:

    def __init__(self, model, data, parameters):
        self.model = model
        self.data = data
        self.parameters = parameters

    def test_sap(self):
        attack_sap = SaltAndPepperNoise(self.parameters)
        result = run_attack(attack_sap, self.model, self.data)
        return result




