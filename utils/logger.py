import logging

log_format = logging.Formatter('%(levelname)s - %(message)s')

handler = logging.StreamHandler()
handler.setFormatter(log_format)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

def input_err_msg(attack: str):
    return f"Input given for running tests does not contain parameters for the {attack} attack test."

def log_attack_start_msg(attack: str):
    logger.info(f"\nPerforming {attack} attack test:")
