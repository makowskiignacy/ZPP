from art.attacks.evasion import DeepFool as DeepFoolOriginal

from attacks.art_attack import ARTAttack


class DeepFool(ARTAttack):
    def __init__(self, **params):
        # Initialization of the arguments needed for the classifier
        super().__init__(**params)

        self._attack_params = {
            "max_iter": 100,
            "epsilon": 1e-06,
            "nb_grads": 10,
            "verbose": True,
        }

        # Assigning only relevant arguments
        for key in self._attack_params.keys():
            if key in params.keys():
                self._attack_params[key] = params[key]

    def conduct(self, model, data):
        self._set_classifier(model)
        self._set_data(data)

        # __init__(classifier: CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE, max_iter: int = 100, epsilon: float = 1e-06,
        # nb_grads: int = 10, batch_size: int = 1, verbose: bool = True)→ None

        # generate(x: ndarray, y: Optional[ndarray] = None, **kwargs)→ ndarray
        return self.to_unified_format(DeepFoolOriginal(classifier=self._classifier, **self._attack_params).generate(
            x=self._data.input, y=self._data.output))