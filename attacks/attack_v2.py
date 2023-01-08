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
    