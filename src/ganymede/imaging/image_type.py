# python
from enum import IntEnum
# 3rd party
from nameof import nameof


class ImageType(IntEnum):
    RGB  = 0
    BGR  = 1
    GRAY = 2

    @staticmethod
    def from_str(s):
        if s == 'rgb':
            return ImageType.RGB
        elif s == 'bgr':
            return ImageType.BGR
        elif s == 'gray':
            return ImageType.GRAY
        else:
            raise ValueError(f'invalid ImageType str:{s}')
            

    def get_channels(self) -> int:
        if   self == ImageType.BGR:  return 3
        elif self == ImageType.RGB:  return 3
        elif self == ImageType.GRAY: return 1
        else: raise ValueError(f'invliad {nameof(ImageType)} value:{int(self)}')