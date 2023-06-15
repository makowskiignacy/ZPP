from attacks.foolboxattacks.carlini_wagner import L2CarliniWagner

from attacks.attack import run_attack
from utils.logger import test_logger
from utils.misc import is_binary_model


class TestCarliniWagner():

    def __init__(self, model, data, parameters):
        self.model = model
        self.data = data
        self.parameters = parameters

    def test_cw_l2(self):
        test_logger.info("L2 Carlini Wagner attack test running")
        attack_an_l2_g = L2CarliniWagner(self.parameters)
        result_an_l2_g = run_attack(attack_an_l2_g, self.model, self.data)
        return result_an_l2_g
