import abc
import torch
import sklearn
from tensorflow import keras

from attacks.attack_v2 import Attack
from art.estimators.classification\
    import KerasClassifier, PyTorchClassifier, SklearnClassifier
# from mlplatformlib.model_building.binary.neural_network\
#     import BinaryNeuralNetworkClassifier

class ARTAttack(Attack):
    # Metoda statyczna służąca do unifikowania formatu danych wyjściowych
    # NOTE aktualnie nie zmienia nic, w przyszłości może znaleźc większe użycie
    @staticmethod
    def to_unified_format(data_from_attack):
        pass

    @abc.abstractmethod
    def __init__(self, **params):
        super().__init__(**params)

    def __set_data(self, data):
        # TODO dorzucić tu lub wyżej sprawdzanie poprawności danych
        self._data = data

    def __set_classifier(self, model):
        if isinstance(model, keras.Model):
            # Setting classifier for Keras model, todo adding optional parameters
            self._classifier = KerasClassifier(model=model)
        # elif isinstance(model, BinaryNeuralNetworkClassifier):
        #     self._classifier = model.get_sklearn_object()
        elif isinstance(model, torch.nn.Module):
            # Pytorch model
            if self.input_shape is None or self.loss is None or self.nb_classes is None:
                raise Exception("PyTorch model needs input_shape, loss and nb_classes to conduct attack")
            self._classifier = PyTorchClassifier(
                model=model, input_shape=self.input_shape, loss=self.loss, nb_classes=self.nb_classes)
        elif isinstance(model, sklearn.base.BaseEstimator):
            # todo adding optional parameters for sklearn model classifier
            self._classifier = SklearnClassifier(model=model)
        else:
            # TODO: Implementation of TensorFlow (and V2) classifiers, BlackBox
            raise Exception("Model type not supported")

    @abc.abstractmethod
    def conduct(self, model, data):
        raise NotImplementedError
        
