# 3rd party
import numpy as np
# project

IMAGE_FILE_FORMATS = [
    'png',
    'jpg',
    'tiff',
    'bmp'
]


def get_channels(img : np.ndarray) -> int:
    if len(img.shape) == 2: return 1
    else: return img.shape[2]


def is_img_file_format(format : str) -> bool:
    format = format.lower()

    return format in IMAGE_FILE_FORMATS
