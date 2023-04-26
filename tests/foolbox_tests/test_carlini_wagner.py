from attacks.foolboxattacks.carlini_wagner import L2CarliniWagner

from attacks.attack import run_attack


class TestL2CarliniWagner():

    def __init__(self, model, data, parameters):
        self.model = model
        self.data = data
        self.parameters = parameters

    def test_cw_l2(self):
        print("L2 Carlini Wagner attack test running")
        attack_an_l2_g = L2CarliniWagner(self.parameters)
        result_an_l2_g = run_attack(attack_an_l2_g, self.model, self.data)
        return result_an_l2_g
