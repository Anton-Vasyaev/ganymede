# python
from enum   import IntEnum, auto
# 3rd party
from nameof import nameof


class ImageType(IntEnum):
    UNKNOWN = auto()

    RGB = auto()

    BGR = auto()

    GRAY = auto()

    RGBA = auto()

    BGRA = auto()


    def get_channels(self) -> int:
        if self == ImageType.BGR:
            return 3
        elif self == ImageType.RGB:
            return 3
        elif self == ImageType.GRAY:
            return 1
        elif self == ImageType.RGBA:
            return 4
        elif self == ImageType.BGRA:
            return 4
        else:
            raise ValueError(f'invalid {nameof(ImageType)} value:{int(self)}')