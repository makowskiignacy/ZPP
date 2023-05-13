"""
Instalacja paczek:
$ pip3 install mlplatformclient-0.0.10.tar.gz
$ pip3 install naskrdsecurity-0.0.4.tar.gz

Przed połączeniem należy dodać certyfikat!!!
$ sudo -iu root
$ mv Internal_CBiTT_CA_.crt /usr/local/share/ca-certificates/
$ update-ca-certificates

Przed uruchomieniem całego skryptu trzeba zaimportować zmienną!!!
$ export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

Pełna dokumentacja znajduje się w pliku __init__.py głównego katalogu paczki.
"""

import time
import os

from mlplatformclient import MLPlatformClient, NextCloud

# Dodać ścieżkę do przykładowego pliku csv na lokalnej maszynie!!!!
EXAMPLE_CSV_FILE_PATH = '/home/grzegorz/priv/mim/zpp/UW_starter_pack/grze.csv'
EXAMPLE_CSV_NAME = 'dataset.csv'
PATH_TO_DOWNLOAD_DIR = 'przykladowy_folder'
NEXT_CLOUD_DIR = '/uw_demo'
# Przykładowy model który zostanie pobrany z ml-platform
EXPERIMENT_NAME = '01339e26-a4b9-4d4b-b999-fd598a59576a'
RUN_NAME = 'skam_search'

class MLPlatformClientConfig:
    ml_platform_address: str = 'ml-platform.test.rd.nask.pl'
    keycloak_address: str = 'auth.rd.nask.pl'
    realm: str = 'ML-PLATFORM'
    client_id: str = 'ml_platform'
    # username: str = 'test'
    # password: str = 'test'
    username: str = 'aml-component'
    password: str = 'QlCsR6FnO2JTx7dqqOCoVJd9'


class NextCloudConfig:
    host: str = 'https://nextcloud.cbitt.nask.pl/'
    # Wstawić własnego usera i hasło!!!
    user: str = 'grzegorzzu'
    password: str = 'p3;.(aM\R4RbqiLwR-.]RQJ5&~Kf||Tx'


ml_p_conf = MLPlatformClientConfig()
next_cloud_conf = NextCloudConfig()


# Tworzenie instancji klienta Ml-platform:
ml_platform = MLPlatformClient(ml_platform_address=ml_p_conf.ml_platform_address,
                               keycloak_address=ml_p_conf.keycloak_address,
                               realm=ml_p_conf.realm,
                               client_id=ml_p_conf.client_id,
                               username=ml_p_conf.username,
                               password=ml_p_conf.password)

# Tworzenie instancji klienta NextCloud:
next_cloud = NextCloud(host=next_cloud_conf.host,
                       user=next_cloud_conf.user,
                       password=next_cloud_conf.password)

# Wrzucanie zbioru do NextCloud
local_file_path = os.path.abspath(EXAMPLE_CSV_FILE_PATH)
next_cloud.upload_file(local_file_path=local_file_path, remote_dir=NEXT_CLOUD_DIR)


# Rejestrowanie zbioru z NextCloud w Ml-platform
nextcloud_source_path = os.path.join(NEXT_CLOUD_DIR, os.path.basename(EXAMPLE_CSV_FILE_PATH))
dataset_info = ml_platform.register_dataset(source_path=nextcloud_source_path)


# Weryfikacja czy zbiór został zarejestrowany
while True:
    time.sleep(2)
    response_details = ml_platform.dataset_details(dataset_id=dataset_info['id'])
    if response_details['uploaded']:
        break