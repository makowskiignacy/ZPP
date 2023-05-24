from cloud_sync import get_model
import numpy as np

EXPERIMENT_NAME = '01339e26-a4b9-4d4b-b999-fd598a59576a'
RUN_NAME = 'skam_search'

def test_model_usage():
    # An example of using a model downloaded from MLPlatform
    model = get_model(EXPERIMENT_NAME, RUN_NAME, './TEMPORARY')
    input_example = np.array(np.random.rand(71), dtype=np.float32)
    print(model.predict(input_example))

if __name__ == "__main__":
    test_model_usage()