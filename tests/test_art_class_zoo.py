# -*- coding: UTF-8 -*-

import unittest

from attacks.attack import Attack

from attacks.art_attack import ARTAttack

from attacks.artattacks.zeroth_order_optimization_bb_attack\
    import ZeorthOrderOptimalization as ZOOAttack

class TestZOO(unittest.TestCase):
    def test_class_hierarchy(self):
        # # Nie powinno być możliwości utworzenia obiektu klasy abstrakcyjnej
        assertion_var = 0
        print("Próba utowrzenia obiektu klasy Attack:\n****")
        try:
            AbsA = Attack()
            assertion_var += 1
            print("Brak błędów.")
        except Exception as e:
            print(e)
        finally:
            print("****\n")

        self.assertEqual(assertion_var, 0)

        print("Próba utowrzenia obiektu klasy ARTAttack:\n****")
        try:
            AbsA = ARTAttack()
            assertion_var += 1
            print("Brak błędów.")
        except Exception as e:
            print(e)
        finally:
            print("****\n")

        self.assertEqual(assertion_var, 0)

        print("Klasę konkretnego ataku można już zinstancjonować:\n****")
        try:
            ZOO_attack = ZOOAttack()
            assertion_var += 1
            print("Brak błędów.")
        except Exception as e:
            print(e)
        finally:
            print("****\n")

        self.assertEqual(assertion_var, 1)

        default_params = ZOO_attack._attack_params

        ZOO_attack = ZOOAttack(confidence=0.50, abort_early=False, max_iter=100, boi_o_boi="YES")
        changed_params = ZOO_attack._attack_params

        self.assertNotIn('boi_o_boi', changed_params.keys())

        for key, val in default_params.items():
            if changed_params[key] != val:
                print("diff:", key, "default:", val, "changed:", changed_params[key])
            else:
                print("same:", key, "=", val)

        print("Zakończono pomyślnie.")

