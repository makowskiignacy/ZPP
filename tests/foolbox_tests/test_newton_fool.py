from attacks.foolboxattacks.newton_fool import NewtonFool

from attacks.attack import run_attack
from utils.logger import test_logger
from utils.misc import is_binary_model


class TestNewtonFool():

    def __init__(self, model, data, parameters):
        self.model = model
        self.data = data
        self.parameters = parameters

    def test_nf(self):
        test_logger.info("Newton Fool attack test running")
        attack_an_nf = NewtonFool(self.parameters)
        result_an_nf = run_attack(attack_an_nf, self.model, self.data)
        return result_an_nf
