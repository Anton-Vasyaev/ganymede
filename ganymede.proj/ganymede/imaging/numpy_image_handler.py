# python
from dataclasses import dataclass
# 3rd party
import numpy as np

from nameof import nameof
# project
from .data.image_type import ImageType
from .auxiliary  import get_channels_of_numpy


class NumpyImageHandler:
    image : np.ndarray

    image_type : ImageType


    def __init__(
        self,
        image      : np.ndarray,
        image_type : ImageType    
    ):
        img_channels      = get_channels_of_numpy(image)
        img_type_channels = image_type.get_channels()

        if img_channels != img_type_channels:
            raise ValueError(
                f'{nameof(NumpyImageHandler)} validation error, image channels != {nameof(ImageType)} channnels:'
                f'{img_channels} != {img_type_channels}.'
            )
        
        self.image      = image
        self.image_type = image_type