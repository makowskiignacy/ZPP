from attacks.foolboxattacks.NewtonFool import NewtonFool
from tests.test_class import Test
# from tests import Test


def main():
    generic_args = {}
    attack_specific_args_simple = {'steps': 100}
    attack_specific_args_nn = {'steps': 100}

    attack_simple = NewtonFool(attack_specific_args_simple, generic_args)
    attack_nn = NewtonFool(attack_specific_args_nn, generic_args)
    testor = Test(attack_simple=attack_simple, attack_nn=attack_nn)
    testor.prep_simple_test(batchsize=4)
    attack_name = "NewtonFool"

    print(f"Attack {attack_name} simple with tiny batchsize")
    result1s = testor.conduct()
    print(f"Attack {attack_name} results:\n{result1s}")

    # testor.prep_simple_test(batchsize=20)
    # print(f"Attack {attack_name} simple with sligtly larger batchsize")
    # result2s = testor.conduct()
    # print(f"Attack {attack_name} results:\n{result2s}")


    # testor.prep_nn_test()
    # if testor.nn_model is not None and testor.nn_data is not None:
    #     print(f"Attack {attack_name} nn")
    #     result1nn = testor.conduct()
    #     print(f"Attack {attack_name} results:\n{result1nn}")


if __name__ == '__main__':
    main()
