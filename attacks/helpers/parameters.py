class Parameter:
    '''
    Klasa abstrakcyjna uogólniająca parametry przekazywane atakom.
    '''
    def update(self, other_parameters):
        raise NotImplementedError
    pass


class FoolboxParameters(Parameter):
    '''
    Klasa realizująca przekazywanie odpowiednich parametrów atakom z bibloteki
    Foolbox.

    Atrybuty obiektu:
    -----------------
    attak_specific_parameters (dict)
        Parametry odpowiednie tylko dla konkretnego ataku.
    generic_parameters (dict)
        Parametry ogólne dla ataków z biblioteki Foolbox.
    '''
    def __init__(self, attack_specific_parameters, generic_parameters):
        self.attack_specific_parameters = attack_specific_parameters
        self.generic_parameters = generic_parameters

    def update(self, other):
        self.attack_specific_parameters.update(other.attack_specific_parameters)
        self.generic_parameters.update(other.generic_parameters)

    def __str__(self):
        return self.attack_specific_parameters.__str__() +\
               self.generic_parameters.__str__()


class ARTParameters(Parameter):
    '''
    Klasa realizująca przekazywanie odpowiednich parametrów atakom z bibloteki
    Adversarial Robustness Toolbox.

    Atrybuty obiektu:
    -----------------
    attack_parameters (dict)
        Parametry odpowiednie tylko dla konkretnego ataku.
    classifier_parameters (dict)
        Parametry odpowiednie dla atakowanego klasyfikatora.
    '''
    def __init__(self, classifier_parameters, attack_parameters):
        self.classifier_parameters=classifier_parameters
        self.attack_parameters=attack_parameters

    def update(self, other):
        self.classifier_parameters.update(other.classifier_parameters)
        self.attack_parameters.update(other.attack_parameters)
    
    def __str__(self):
        return self.classifier_parameters.__str__() +\
               self.attack_parameters.__str__()