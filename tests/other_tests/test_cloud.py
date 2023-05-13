from cloud_sync import get_model
from utils.logger import test_logger


def test_cloud_download():
    # Sample model to be downloaded from ml-platform
    EXPERIMENT_NAME = '01339e26-a4b9-4d4b-b999-fd598a59576a'
    RUN_NAME = 'skam_search'
    test_logger.info(get_model(EXPERIMENT_NAME, RUN_NAME, './TEMPORARY'))


def test_cloud_upload():
    pass

if __name__ == "__main__":
    test_cloud_download()
