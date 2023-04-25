from foolbox.models.pytorch import PyTorchModel
import time
from attacks.helpers.data import Data
from foolbox.utils import accuracy

# Chcemy mieć możliwość określenia metody jako abstrakcyjnej
import abc

# Chcemy mieć możliwość określenia klasy jako abstrakcyjnej
from abc import ABC


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


def run_attack(attack: Attack, model, data: Data):
    time_start = time.time()

    # print(type(model))
    if isinstance(model, PyTorchModel):
        print(f"Model accuracy before attack: {accuracy(model, data.input, data.output)}")
    print(f"Starting attack. ({time.asctime(time.localtime(time_start))})")

    adversarials = attack.conduct(model, data)

    time_end = time.time()
    print(f"Attack done. ({time.asctime(time.localtime(time_end))})")
    print(f"Took {time_end - time_start}\n")

    if adversarials is not None and isinstance(model, PyTorchModel):
        print(f"Model accuracy after attack: {accuracy(model, adversarials, data.output)}")

    return adversarials