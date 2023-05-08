import os.path
import time
import traceback
import sys
import shutil
from mlplatformclient import MLPlatformClient
from zipfile import ZipFile
# https://www.mlflow.org/docs/latest/models.html
import mlflow
# https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html#module-mlflow.sklearn
import mlflow.sklearn

class MLPlatformClientConfig:
    ml_platform_address: str = 'ml-platform.test.rd.nask.pl'
    keycloak_address: str = 'auth.rd.nask.pl'
    realm: str = 'ML-PLATFORM'
    client_id: str = 'ml_platform'
    username: str = 'test'
    password: str = 'test'

def download_compressed_model(experiment_name, run_name, download_dir):

    ml_p_conf = MLPlatformClientConfig()

    # Tworzenie instancji klienta Ml-platform:
    ml_platform = MLPlatformClient(ml_platform_address=ml_p_conf.ml_platform_address,
                                   keycloak_address=ml_p_conf.keycloak_address,
                                   realm=ml_p_conf.realm,
                                   client_id=ml_p_conf.client_id,
                                   username=ml_p_conf.username,
                                   password=ml_p_conf.password)
        
    # Pobieranie artefaktów z modelem
    return ml_platform.get_compressed_artifacts(mlflow_experiment_name=experiment_name,
                                                           mlflow_run_name=run_name,
                                                           zip_destination_path=download_dir)


def get_model(experiment_name, run_name, download_dir):
    if os.path.isdir(download_dir):
        raise NameError("Download directory already exist")
    
    model_description = download_compressed_model(experiment_name, run_name, download_dir)

    with ZipFile(model_description['file_path'],"r") as zip_ref:
        zip_ref.extractall(download_dir)

    pipeline = mlflow.sklearn.load_model(download_dir)
    standard_scaler = pipeline.steps[0][1]
    shutil.rmtree(download_dir)
    return pipeline.steps[1][1]