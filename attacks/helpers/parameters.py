class Parameter:
    pass

class FoolboxParameters(Parameter):
    def __init__(self, attack_specific_parameters, generic_parameters):
        self.attack_specific_parameters = attack_specific_parameters
        self.generic_parameters = generic_parameters

class ARTParameters(Parameter):
    def __init__(self, classifier_parameters, attack_parameters):
        self.classifier_parameters=classifier_parameters
        self.attack_parameters=attack_parameters