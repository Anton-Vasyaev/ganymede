# python
from enum import IntEnum, auto


class NmsKindType(IntEnum):
    DEFAULT_NMS = auto() 
    GREEDY_NMS = auto() 
    DIOU_NMS = auto() 
    CORNERS_NMS = auto()

    @staticmethod
    def from_str(data : str):
        if data == 'default': return NmsKindType.DEFAULT_NMS
        elif data == 'greedynms': return NmsKindType.GREEDY_NMS
        elif data == 'diounms': return NmsKindType.DIOU_NMS
        elif data == 'corners': return NmsKindType.CORNERS_NMS
        else:
            raise ValueError(f'Can not parse NMS kind type:{data}')