# -*- coding: UTF-8 -*-

import unittest



from attacks.art_attack import ARTAttack
from attacks.attack import Attack

from attacks.artattacks.zeroth_order_optimization_bb_attack\
    import ZerothOrderOptimalization as ZOOAttack
from attacks.helpers.parameters import ARTParameters
from utils.logger import test_logger

class TestClassHierarchyArt(unittest.TestCase):
    def test_class_hierarchy(self):
        # # Nie powinno być możliwości utworzenia obiektu klasy abstrakcyjnej
        assertion_vars = []
        assertion_var = 0
        test_logger.info("Próba utworzenia obiektu klasy Attack:\n****")
        try:
            AbsA = Attack()
            assertion_var += 1
            test_logger.info("Brak błędów.")
        except Exception as e:
            test_logger.info(e)
        finally:
            test_logger.info("****\n")

        assertion_vars.append(assertion_var == 0)

        test_logger.info("Próba utworzenia obiektu klasy ARTAttack:\n****")
        try:
            AbsA = ARTAttack({})
            assertion_var += 1
            test_logger.info("Brak błędów.")
        except Exception as e:
            test_logger.error(e)
        finally:
            test_logger.info("****\n")

        assertion_vars.append(assertion_var == 0)

        test_logger.info("Klasę konkretnego ataku można już zinstancjonować:\n****")
        try:
            ZOO_attack = ZOOAttack(ARTParameters({}, {}))
            assertion_var += 1
            test_logger.info("Brak błędów.")
        except Exception as e:
            test_logger.error(e)
        finally:
            test_logger.info("****\n")

        assertion_vars.append(assertion_var == 1)

        default_params = ZOO_attack._attack_params

        attack_parameters={"confidence": 0.5, "abort_early": False, "max_iter": 100}

        ZOO_attack = ZOOAttack(ARTParameters(classifier_parameters={}, attack_parameters=attack_parameters))
        changed_params = ZOO_attack._attack_params


        for key, val in default_params.items():
            if changed_params[key] != val:
                test_logger.info(f"diff: {key}, default: {val}, changed: {changed_params[key]}")
            else:
                test_logger.info(f"same: {key} = {val}")

        test_logger.info("Zakończono pomyślnie.")
        return assertion_vars

