from attacks.foolboxattacks.NewtonFool import NewtonFool
import tests
# from tests import Test


def main():
    attack_specific_args_simple = {"steps": 10, "across_channels": False}
    generic_args = {}

    attack_specific_args_nn = {"steps": 1000, "across_channels": False}

    attack_simple = NewtonFool(attack_specific_args_simple, generic_args)
    attack_nn = NewtonFool(attack_specific_args_nn, generic_args)
    testor = tests.Test(attack_simple=attack_simple, attack_nn=attack_nn)
    smodel, sdata = testor.prep_simple_test(batchsize=4)
    attack_name = "LBFGS"

    print(f"Attack {attack_name} simple with tiny batchsize")
    result1s = testor.conduct(attack_simple, smodel, sdata)
    print(f"Attack {attack_name} results:\n{result1s}")

    smodel, sdata = testor.prep_simple_test(batchsize=20)
    print(f"Attack {attack_name} simple with sligtly larger batchsize")
    result2s = testor.conduct(attack_simple, smodel, sdata)
    print(f"Attack {attack_name} results:\n{result2s}")


    nn_model, nn_data = testor.prep_nn_test()
    if nn_model is not None and nn_data is not None:
        print(f"Attack {attack_name} nn")
        result1nn = testor.conduct(attack_nn, nn_model, nn_data)
        print(f"Attack {attack_name} results:\n{result1nn}")


if __name__ == '__main__':
    main()
