# 3rd party
import numpy as np
# project

IMAGE_FILE_FORMATS = [
    'png',
    'jpg',
    'tiff',
    'bmp'
]


def get_channels_of_numpy(img : np.ndarray) -> int:
    if len(img.shape) == 2: return 1
    else: return img.shape[2]


def get_channels(img : np.ndarray) -> int:
    return get_channels_of_numpy(img)


def get_row_stride_of_numpy(img : np.ndarray) -> int:
    h, w = img.shape[0:2]

    arr_info = img.__array_interface__
    if arr_info['strides'] is None:
        return w * img.dtype.itemsize
    
    return arr_info['strides'][0]



def is_img_file_format(format : str) -> bool:
    format = format.lower()

    return format in IMAGE_FILE_FORMATS
