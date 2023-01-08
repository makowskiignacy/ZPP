import abc
import torch
import sklearn
from tensorflow import keras

from attacks.attack_v2 import Attack
from art.estimators.classification \
    import KerasClassifier, PyTorchClassifier, SklearnClassifier
from mlplatformlib.model_building.binary.neural_network \
    import BinaryNeuralNetworkClassifier


class ARTAttack(Attack):
    # Metoda statyczna służąca do unifikowania formatu danych wyjściowych
    # NOTE aktualnie nie zmienia nic, w przyszłości może znaleźc większe użycie
    @staticmethod
    def to_unified_format(data_from_attack):
        pass

    @abc.abstractmethod
    def __init__(self, **params):
        super().__init__()
        # Parameters required to initialize the classifier
        self._attack_params = {
            'mask': None,
            'reset_patch': False,
            'input_shape': None,
            'loss': None,
            'nb_classes': None,
            'optimizer': None,
            'clip_values': None
        }

    def _set_data(self, data):
        # TODO dorzucić tu lub wyżej sprawdzanie poprawności danych
        self._data = data

    def _set_classifier(self, model):
        if isinstance(model, keras.Model):
            # Setting classifier for Keras model, todo adding optional parameters
            self._classifier = KerasClassifier(model=model)
        elif isinstance(model, BinaryNeuralNetworkClassifier):
            self._classifier = model.get_sklearn_object()
        elif isinstance(model, torch.nn.Module):
            # Pytorch model
            if self._attack_params.get('input_shape') and self._attack_params.get('loss') and self._attack_params.get('nb_classes'):
                self._classifier = PyTorchClassifier(model=model, **{key: self._attack_params.get(key) for key in ['loss', 'optimizer', 'input_shape', 'nb_classes', 'clip_values']})
            else:
                raise Exception("PyTorch model needs input_shape, loss and nb_classes to conduct attack")
        elif isinstance(model, sklearn.base.BaseEstimator):
            # todo adding optional parameters for sklearn model classifier
            self._classifier = SklearnClassifier(model=model)
        else:
            # TODO: Implementation of TensorFlow (and V2) classifiers, BlackBox
            raise Exception("Model type not supported")

    @abc.abstractmethod
    def conduct(self, model, data):
        raise NotImplementedError