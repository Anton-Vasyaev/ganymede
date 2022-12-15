# python
from enum import IntEnum, auto


class AugmentationType(IntEnum):
    BASIC_COLOR  = auto()
    MIRROR       = auto()
    PADDING      = auto()
    ROTATE_2D    = auto()
    ROTATE_3D    = auto()
    STRETCH      = auto()