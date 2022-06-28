# python
from enum import IntEnum

class StretchType(IntEnum):
    SRC = 0
    DST = 1

    def from_str(str):
        if   str == 'src': return StretchType.SRC
        elif str == 'dst': return StretchType.DST
        else: raise Exception(f'invalid str:{str}')