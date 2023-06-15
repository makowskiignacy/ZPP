from torch import Tensor


def is_binary_model(targets: Tensor):
    return targets.shape[1] == 1
