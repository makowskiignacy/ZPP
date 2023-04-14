from attacks.foolboxattacks.basic_iterative import L1BasicIterative, L2BasicIterative, LinfBasicIterative
from attacks.foolboxattacks.basic_iterative import L1AdamBasicIterative, L2AdamBasicIterative, LinfAdamBasicIterative

from attacks.attack import run_attack


class TestBasicIterative:

    def __init__(self, model, data, parameters):
        self.model = model
        self.data = data
        self.parameters = parameters

    def test_bi_1(self):
        print("L1 Basic Iterative attack test running")
        attack_bi1 = L1BasicIterative(self.parameters)
        result1 = run_attack(attack_bi1, self.model, self.data)
        return result1

    def test_bi_2(self):
        print("L2 Basic Iterative attack test running")
        attack_bi2 = L2BasicIterative(self.parameters)
        result2 = run_attack(attack_bi2, self.model, self.data)
        return result2

    def test_bi_inf(self):
        print("L infinity Basic Iterative attack test running")
        attack_bi_inf = LinfBasicIterative(self.parameters)
        resultinf = run_attack(attack_bi_inf, self.model, self.data)
        return resultinf

    def test_bi_1_a(self):
        print("L1 Basic Iterative attack with Adam optimizer test running")
        attack_bi1_a = L1AdamBasicIterative(self.parameters)
        result1a = run_attack(attack_bi1_a, self.model, self.data)
        return result1a

    def test_bi_2_a(self):
        print("L2 Basic Iterative attack with Adam optimizer test running")
        attack_bi2_a = L2AdamBasicIterative(self.parameters)
        result2a = run_attack(attack_bi2_a, self.model, self.data)
        return result2a

    def test_bi_inf_a(self):
        print("L infinity Basic Iterative attack with Adam optimizer test running")
        attack_bi_inf_a = LinfAdamBasicIterative(self.parameters)
        resultinfa = run_attack(attack_bi_inf_a, self.model, self.data)
        return resultinfa


