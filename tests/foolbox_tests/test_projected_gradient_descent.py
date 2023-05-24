from attacks.foolboxattacks.projected_gradient_descent import L1ProjectedGradientDescent, L2ProjectedGradientDescent, LinfProjectedGradientDescent
from attacks.foolboxattacks.projected_gradient_descent import L1AdamProjectedGradientDescent, L2AdamProjectedGradientDescent, LinfAdamProjectedGradientDescent
from attacks.attack import run_attack
from utils.logger import test_logger

class TestProjectedGradientDescent:
    def __init__(self, model, data, parameters):
        self.model = model
        self.data = data
        self.parameters = parameters

    def test_pgd_1(self):
        test_logger.info("L1 Projected Gradient Descent attack test running")
        attack_pgd1 = L1ProjectedGradientDescent(self.parameters)
        result1 = run_attack(attack_pgd1, self.model, self.data)
        return result1

    def test_pgd_2(self):
        test_logger.info("L2 Projected Gradient Descent attack test running")
        attack_pgd2 = L2ProjectedGradientDescent(self.parameters)
        result2 = run_attack(attack_pgd2, self.model, self.data)
        return result2

    def test_pgd_inf(self):
        test_logger.info("L infinity Projected Gradient Descent attack test running")
        attack_pgdinf = LinfProjectedGradientDescent(self.parameters)
        resultinf = run_attack(attack_pgdinf, self.model, self.data)
        return resultinf

    def test_pgd_1_a(self):
        test_logger.info("L1 Projected Gradient Descent attack with Adam optimizer test running")
        attack_pgd1_a = L1AdamProjectedGradientDescent(self.parameters)
        result1a = run_attack(attack_pgd1_a, self.model, self.data)
        return result1a

    def test_pgd_2_a(self):
        test_logger.info("L2 Projected Gradient Descent attack with Adam optimizer test running")
        attack_pgd2_a = L2AdamProjectedGradientDescent(self.parameters)
        result2a = run_attack(attack_pgd2_a, self.model, self.data)
        return result2a

    def test_pgd_inf_a(self):
        test_logger.info("L infinity Projected Gradient Descent attack with Adam optimizer test running")
        attack_pgdinf_a = LinfAdamProjectedGradientDescent(self.parameters)
        resultinfa = run_attack(attack_pgdinf_a, self.model, self.data)
        return resultinfa



