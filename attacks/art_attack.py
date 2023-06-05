import abc

import sklearn
from eagerpy.astensor import astensor
from skorch import NeuralNetBinaryClassifier
import torch
from art.estimators.classification \
    import KerasClassifier, PyTorchClassifier, SklearnClassifier
from mlplatformlib.model_building.binary.neural_network \
    import BinaryNeuralNetworkClassifier
from tensorflow import keras

from attacks.attack import Attack

import art
class ARTAttack(Attack):
    # Metoda statyczna służąca do unifikowania formatu danych wyjściowych
    # NOTE aktualnie nie zmienia nic, w przyszłości może znaleźc większe użycie
    @staticmethod
    def to_unified_format(data_from_attack):
        # Not implemented yet
        return data_from_attack

    @abc.abstractmethod
    def __init__(self, params):
        super().__init__()

        # Parameters required to initialize the classifier
        self._classifier_params = {
            'mask': params.get('mask'),
            'reset_patch': params.get('reset_patch', False),
            'input_shape': params.get('input_shape'),
            'loss': params.get('loss'),
            'nb_classes': params.get('nb_classes'),
            'optimizer': params.get('optimizer'),
            'clip_values': params.get('clip_values'),
            'use_logits': params.get('use_logits', False)
        }

    def _set_data(self, data):
        # TODO dorzucić tu lub wyżej sprawdzanie poprawności danych
        self._data = data

    def _set_classifier(self, model, data):
        self.verify_clip_values(data)
        if isinstance(model, keras.Model):
            # Setting classifier for Keras model, todo adding optional parameters
            self._classifier = KerasClassifier(
                model=model, **{key: self._classifier_params.get(key) for key in
                                ['clip_values', 'use_logits']})
        elif isinstance(model, torch.nn.Module):
            # Pytorch model
            if self._classifier_params.get('input_shape')\
                and self._classifier_params.get('loss')\
                and self._classifier_params.get('nb_classes'):
                self._classifier = PyTorchClassifier(
                    model=model, **{key: self._classifier_params.get(key) for key in
                                    ['loss', 'optimizer', 'input_shape', 'nb_classes', 'clip_values']})
            else:
                raise Exception("PyTorch model needs input_shape, loss and nb_classes to conduct attack")
        elif isinstance(model, BinaryNeuralNetworkClassifier):
            self._classifier = model.get_sklearn_object()
        elif isinstance(model, NeuralNetBinaryClassifier):
            self._classifier = PyTorchClassifier(
                model=model.module_,
                **{key: self._classifier_params.get(key) for key in
                                    ['loss', 'optimizer', 'input_shape', 'nb_classes', 'clip_values']}
            )
        elif isinstance(model, sklearn.base.BaseEstimator):
            # todo adding optional parameters for sklearn model classifier
            self._classifier = SklearnClassifier(model=model)
        else:
            # TODO: Implementation of TensorFlow (and V2) classifiers, BlackBox
            raise Exception("Model type not supported\n" + str(self.__class__.mro()) + "\n")

    @abc.abstractmethod
    def conduct(self, model, data):
        raise NotImplementedError

    def verify_clip_values(self, data):
        if self._classifier_params.get('clip_values') is not None:
            return
        input_values = astensor(data.input)
        self._classifier_params['clip_values'] = (input_values.min().item(), input_values.max().item())
