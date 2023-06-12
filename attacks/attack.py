from foolbox.models.pytorch import PyTorchModel
from art.estimators.classification import PyTorchClassifier
import time
from attacks.helpers.data import Data
from foolbox.utils import accuracy

# Chcemy mieć możliwość określenia metody jako abstrakcyjnej
import abc

# Chcemy mieć możliwość określenia klasy jako abstrakcyjnej
from abc import ABC
from utils.logger import test_logger

# *Abstrakcyjna* klasa będąca uogólnionym atakiem, którym można zaatakować
# wybrany model o określonym zbiorze danych wejściowych
class Attack(ABC):
    # Tworzy obiekt ataku ustawiając jego podstawowe parametry
    @abc.abstractmethod
    def __init__(self):
        # Definiujemy zmienne obiektu, jakich będziemy używać
        self._model = None
        self._data = None
        self._attack_params = {}

        # Oczywiście ta procedura ma być właściwie napisana w konkretnej klasie
        # implementującej atak

    # Przeprowadza atak opisany podczas konstrukcji obiektu na zadanym modelu
    # oraz danych wejściowych
    @abc.abstractmethod
    def conduct(self, model, data):
        raise NotImplementedError

    @abc.abstractmethod
    def accuracy(self, model, input, output):
        raise NotImplementedError


def run_attack(attack: Attack, model, data: Data):
    time_start = time.time()

    # logger.info(type(model))
    test_logger.info(f"Model accuracy before attack: {attack.accuracy(model, data.input, data.output)}")

    test_logger.info(f"Starting attack. ({time.asctime(time.localtime(time_start))})")

    adversarials = attack.conduct(model, data)

    time_end = time.time()
    test_logger.info(f"Attack done. ({time.asctime(time.localtime(time_end))})")
    test_logger.info(f"Took {time_end - time_start}")

    if adversarials is not None:
        test_logger.info(f"Model accuracy after attack: {attack.accuracy(model, adversarials, data.output)}")
    else:
        test_logger.info(f"Attack not successfull, adversarials:\n{adversarials}")
    test_logger.info("\n")

    return adversarials