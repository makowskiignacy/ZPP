# Przykładowe użycie dokumentacji wewnątrz Pythona
from attacks import *
from attacks.art_attack import *
from attack_manager import *

# Dokumentacja klasy abstrakcyjnej ataku
help(Attack)

# oraz jej podklasy specyficznej dla biblioteki
help(ARTAttack)

# Dokumentacja jednego z ataków z biblioteki ART
help(AdversarialPatch)

# Dokumentacja jednego z ataków z biblioteki Foolbox
help(DeepFool)

# Dokumentacja Attack Managera
help(AttackManager)
