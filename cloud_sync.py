"""
Be sure to add the certificate before connecting:
$ sudo -iu root
$ mv Internal_CBiTT_CA_.crt /usr/local/share/ca-certificates/
$ update-ca-certificates

Remember to import the variable before running the script:
$ export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
"""

import os.path
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
    """
    Downloads a compressed model from the ML Platform.

    Parameters:
        - experiment_name (str): The name of the MLflow experiment containing the model.
        - run_name (str): The name of the MLflow run containing the model.
        - download_dir (str): The directory where the downloaded artifact will be saved.

    Returns:
        Dictionary containing model decription.
    """
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
    """
    Downloads a compressed model artifact from the ML Platform, extracts it, and loads the model.

    Parameters:
        - experiment_name (str): The name of the MLflow experiment containing the model.
        - run_name (str): The name of the MLflow run containing the model.
        - download_dir (str): Temporary working directory

    Returns:
        The loaded model.
    """
    if os.path.isdir(download_dir):
        raise NameError("Download directory already exist")
    
    model_description = download_compressed_model(experiment_name, run_name, download_dir)

    with ZipFile(model_description['file_path'],"r") as zip_ref:
        zip_ref.extractall(download_dir)

    pipeline = mlflow.sklearn.load_model(download_dir)
    standard_scaler = pipeline.steps[0][1]

    shutil.rmtree(download_dir)

    return pipeline.steps[1][1]