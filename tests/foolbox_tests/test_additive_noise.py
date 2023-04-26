from attacks.foolboxattacks.additive_noise import L2AdditiveGaussianNoise, L2AdditiveUniformNoise, L2ClippingAwareAdditiveGaussianNoise, L2ClippingAwareAdditiveUniformNoise, L2ClippingAwareRepeatedAdditiveGaussianNoise, L2ClippingAwareRepeatedAdditiveUniformNoise, L2RepeatedAdditiveGaussianNoise, L2RepeatedAdditiveUniformNoise, LinfAdditiveUniformNoise, LinfRepeatedAdditiveUniformNoise

from attacks.attack import run_attack


class TestAdditiveNoise:

    def __init__(self, model, data, parameters):
        self.model = model
        self.data = data
        self.parameters = parameters

    def test_an_l2_g(self):
        print("L2 Additive Gaussian Noise attack test running")
        attack_an_l2_g = L2AdditiveGaussianNoise(self.parameters)
        result_an_l2_g = run_attack(attack_an_l2_g, self.model, self.data)
        return result_an_l2_g

    def test_an_l2_u(self):
        print("L2 Additive Uniform Noise attack test running")
        attack_an_l2_u = L2AdditiveUniformNoise(self.parameters)
        result_an_l2_u = run_attack(attack_an_l2_u, self.model, self.data)
        return result_an_l2_u

    def test_an_l2_cag(self):
        print("L2 Clipping Aware Additive Gaussian Noise attack test running")
        attack_an_l2_cag = L2ClippingAwareAdditiveGaussianNoise(self.parameters)
        result_an_l2_cag = run_attack(attack_an_l2_cag, self.model, self.data)
        return result_an_l2_cag

    def test_an_l2_cau(self):
        print("L2 Clipping Aware Additive Uniform Noise attack test running")
        attack_an_l2_cau = L2ClippingAwareAdditiveUniformNoise(self.parameters)
        result_an_l2_cau = run_attack(attack_an_l2_cau, self.model, self.data)
        return result_an_l2_cau

    def test_an_l2_carg(self):
        print("L2 Clipping Aware Repeated Additive Gaussian Noise attack test running")
        attack_an_l2_carg = L2ClippingAwareRepeatedAdditiveGaussianNoise(self.parameters)
        result_an_l2_carg = run_attack(attack_an_l2_carg, self.model, self.data)
        return result_an_l2_carg

    def test_an_l2_caru(self):
        print("L2 Clipping Aware Repeated Additive Uniform Noise attack test running")
        attack_an_l2_caru = L2ClippingAwareRepeatedAdditiveUniformNoise(self.parameters)
        result_an_l2_caru = run_attack(attack_an_l2_caru, self.model, self.data)
        return result_an_l2_caru

    def test_an_l2_rg(self):
        print("L2 Repeated Additive Gaussian Noise attack test running")
        attack_an_l2_rg = L2RepeatedAdditiveGaussianNoise(self.parameters)
        result_an_l2_rg = run_attack(attack_an_l2_rg, self.model, self.data)
        return result_an_l2_rg

    def test_an_l2_ru(self):
        print("L2 Repeated Additive Uniform Noise attack test running")
        attack_an_l2_ru = L2RepeatedAdditiveUniformNoise(self.parameters)
        result_an_l2_ru = run_attack(attack_an_l2_ru, self.model, self.data)
        return result_an_l2_ru

    def test_an_inf_u(self):
        print("Linf Additive Uniform Noise attack test running")
        attack_an_inf_u = LinfAdditiveUniformNoise(self.parameters)
        result_an_inf_u = run_attack(attack_an_inf_u, self.model, self.data)
        return result_an_inf_u

    def test_an_inf_ru(self):
        print("Linf Repeated Additive Uniform Noise attack test running")
        attack_an_inf_ru = LinfRepeatedAdditiveUniformNoise(self.parameters)
        result_an_inf_ru = run_attack(attack_an_inf_ru, self.model, self.data)
        return result_an_inf_ru
