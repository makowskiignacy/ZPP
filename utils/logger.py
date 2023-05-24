import logging

from utils.config import LOGGING_LVL, LOGGING_FORMAT

log_format = logging.Formatter(LOGGING_FORMAT)
handler = logging.StreamHandler()
test_logger = logging.getLogger('test')
test_logger.propagate = False

handler.setFormatter(log_format)
test_logger.setLevel(LOGGING_LVL)
test_logger.addHandler(handler)

def input_err_msg(attack: str):
    return f"Input given for running tests does not contain parameters for the {attack} attack test."

def log_attack_start_msg(attack: str):
    test_logger.info(f"Performing {attack} attack test:")
