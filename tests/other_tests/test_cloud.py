from cloud_sync import get_model

def test_cloud():
    # Sample model to be downloaded from ml-platform
    EXPERIMENT_NAME = '01339e26-a4b9-4d4b-b999-fd598a59576a'
    RUN_NAME = 'skam_search'
    print(get_model(EXPERIMENT_NAME, RUN_NAME, './TEMPORARY'))

if __name__ == "__main__":
    test_cloud()