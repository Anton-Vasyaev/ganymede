# 3rd party
import numpy as np

from nameof import nameof


def number_of_dimensions_equal(
    arr      : np.ndarray,
    count    : int,
    arr_name : str = ''
):
    arr_dim_count = len(arr.shape)
    if arr_dim_count != count:
        raise ValueError(
            f'number of dimensions of array \'{arr_name}\' is not equal {count}, gotted:{arr_dim_count}.'
        )
    

def is_float32(
    arr      : np.ndarray,
    arr_name : str = ''
):
    if arr.dtype != np.float32:
        raise ValueError(
            f'data type of array \'{arr_name}\' is not equal {nameof(np.float32)}, gotted:{arr.dtype}.'
        )