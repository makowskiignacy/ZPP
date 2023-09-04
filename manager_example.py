# Skrypt pokazujący wykorzytsanie i działanie pełnego modułu.
# Przedstawia: Logowanie, Pobieranie danych i modelu, Utworzenie ataków
#              Uruchomienie ataków

import getpass

from attack_manager import AttackManager

from cloud_sync import get_model
from utils.dataloader import DataLoader
from attacks.helpers.parameters import ARTParameters, FoolboxParameters

from torch import nn, optim

# Docelowo taki skrypt może być zwinięty do pojedynczej procedury, która
# otrzyma tylko potrzebne dane.

### Dane wstępne ###
NEXT_CLOUD_DIR = '/uw_demo'

EXPERIMENT_NAME = '266e5e3e-0a22-4b92-8d70-960718a3b600'
RUN_NAME = 'skam_search'
DATA_ID = None #'997e79b1919a46fbb8d71bdafaf4a8ad'
DATA_FILE_NAME = 'demo_dataset.csv'

nc_user = input("NextCloud username:")
nc_pass = getpass.getpass("NextCloud password:")

dl = DataLoader(
    ml_user='test', ml_pass='test',
    nc_user=nc_user, nc_pass=nc_pass,
    ml_platform_address = 'ml-platform.test.rd.nask.pl',
    ml_keycloak_address = 'auth.rd.nask.pl',
    ml_realm  = 'ML-PLATFORM',
    ml_client_id  = 'ml_platform',
    nc_host  = 'https://nextcloud.cbitt.nask.pl/'
)
# Zmieniamy docelowy zdalny katalog
dl.UPLOAD_DIRECTORY = NEXT_CLOUD_DIR

### Pobieranie danych po ID lub po ścieżce do NextCloud
if (DATA_ID is not None):
    data_info = dl.download_by_id(DATA_ID, 'dataset_' + DATA_ID + '.csv')
    if ('file_path' not in data_info.keys()):
        print(data_info['message'])
        exit(1)
    else:
        data_path = data_info['file_path']
else:
    # Ułatwienie tworzenia ścieżki do pliku w zdalnym katalogu
    remote_path = dl.make_remote_path(DATA_FILE_NAME)
    dl.download(remote_path)
    # oraz do ścieżki do pobranego pliku
    data_path = dl.make_local_path(DATA_FILE_NAME)

data = dl.load_to_variable(
    libs=[DataLoader.SupportedLibrary.ART],
    local_file_path=data_path,
    number_of_samples=1000
)[0]

model = get_model(EXPERIMENT_NAME, RUN_NAME, './TEMPORARY')

art_criterion = nn.CrossEntropyLoss()
art_optimizer = optim.Adam(model.module_.parameters(), lr=0.01)



additional_classifier_parameters = {"loss": art_criterion,
                                    "optimizer" : art_optimizer,
                                    "input_shape": data.input.shape,
                                    "nb_classes": int(data.output.max() - min(0, data.output.min()) + 1)}

params = ARTParameters(additional_classifier_parameters, {})

### Uruchamianie poprzez AttackManager ###

BAD_ATTACK_LIST = [
    ("Jacobian Saliency Map", params),
    ("Some Strange Attack", params),
    ("Square", params)
]

ATTACK_LIST = [
    ("Jacobian Saliency Map",  params),
    ("Square", params)
]

manager = AttackManager()

# Wyjątek AttackNameNotRecognized kiedy podany został nieznany atak
try:
    attacks = manager.create_attacks(BAD_ATTACK_LIST)
except Exception as e:
    print(f"{type(e).__name__}Exception: {e}")

attacks = manager.create_attacks(ATTACK_LIST)

results = manager.conduct_attacks(attacks, model, data)

print(results)

# Usuwamy pobrane pliki danych z systemu, można to zrobić wcześniej
dl.delete_downloaded()