from attacks.foolboxattacks.brendel_bethge import L0BrendelBethge, L1BrendelBethge, L2BrendelBethge, LinfinityBrendelBethge

from attacks.attack import run_attack


class TestBrendelBethge:
    def __init__(self, model, data, parameters):
        self.model = model
        self.data = data
        self.parameters = parameters

    def test_bb_0(self):
        print("L0 Brendel-Bethge attack test running")
        attack_bb0 = L0BrendelBethge(self.parameters)
        result0s = run_attack(attack_bb0, self.model, self.data)
        return result0s

    def test_bb_1(self):
        print("L1 Brendel-Bethge attack test running")
        attack_bb1 = L1BrendelBethge(self.parameters)
        result1s = run_attack(attack_bb1, self.model, self.data)
        return result1s

    def test_bb_2(self):
        print("L2 Brendel-Bethge attack test running")
        attack_bb2 = L2BrendelBethge(self.parameters)
        result2s = run_attack(attack_bb2, self.model, self.data)
        return result2s

    def test_bb_inf(self):
        print("L infinity Brendel-Bethge attack test running")
        attack_bbinf = LinfinityBrendelBethge(self.parameters)
        resultinfs = run_attack(attack_bbinf, self.model, self.data)
        return resultinfs


