# python
from enum import IntEnum, auto


class object_type(IntEnum): # int32_t
    unknown             = auto()
    detections_batch    = auto()