# 3rd party
import numpy as np

def normalize_min_max(array):
    array = array.copy()

    min_v = np.min(array)
    max_v = np.max(array)
    size  = max_v - min_v

    array -= min_v
    array /= max_v

    return array


def inverse_range(array):
    min_v = np.min(array)
    max_v = np.max(array)

    dist = array - min_v

    result = max_v - dist

    return result