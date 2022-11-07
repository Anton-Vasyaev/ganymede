# python
from enum import IntEnum

class StretchOrientation(IntEnum):
    HORIZONTAL = 0
    VERTICAL   = 1

    def from_str(str):
        if   str == 'horizontal': return StretchOrientation.HORIZONTAL
        elif str == 'vertical':   return StretchOrientation.VERTICAL
        else: raise Exception('invalid str:{str}')