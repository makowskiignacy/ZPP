from torch import Tensor


def is_binary_model(targets: Tensor):
    assert len(targets.shape) == 2
    return targets.shape[1] == 1