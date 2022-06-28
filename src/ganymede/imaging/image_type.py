# python
from enum import IntEnum
# 3rd party
from nameof import nameof


class ImageType(IntEnum):
    RGB  = 0
    BGR  = 1
    GRAY = 2


    def get_channels(self) -> int:
        if   self == ImageType.BGR:  return 3
        elif self == ImageType.RGB:  return 3
        elif self == ImageType.GRAY: return 1
        else: raise ValueError(f'invliad {nameof(ImageType)} value:{int(self)}')