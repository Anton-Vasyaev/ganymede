# python
from enum import IntEnum, auto

class OpenCVWindowKey(IntEnum):
    UNKNOWN = 0

    A = auto()
    B = auto() 
    C = auto() 
    D = auto() 
    E = auto()
    F = auto()
    G = auto()
    H = auto()
    I = auto()
    J = auto()
    K = auto()
    L = auto()
    M = auto()
    N = auto()
    O = auto()
    P = auto()
    Q = auto()
    R = auto()
    S = auto()
    T = auto()
    U = auto()
    V = auto()
    W = auto()
    X = auto()
    Y = auto()
    Z = auto()
    
    ESCAPE    = auto() # 27
    SPACE     = auto() # 32
    TILDA     = auto() # 96: ~ `
    TAB       = auto() # 9
    ENTER     = auto() # 13
    BACKSPACE = auto() # 8
    SLASHES   = auto() # | \ /


    @staticmethod
    def from_keycode(code : int):
        if code >= 65 and code <= 90:
            code -= 64

            return OpenCVWindowKey(code)
        elif code >= 97 and code <= 122:
            code -= 96

            return OpenCVWindowKey(code)
        elif code == 27:
            return OpenCVWindowKey.ESCAPE
        elif code == 32:
            return OpenCVWindowKey.SPACE
        elif code == 96:
            return OpenCVWindowKey.TILDA
        elif code == 9:
            return OpenCVWindowKey.TAB
        elif code == 13:
            return OpenCVWindowKey.ENTER
        elif code == 8:
            return OpenCVWindowKey.BACKSPACE
        elif code == 96:
            return OpenCVWindowKey.SLASHES
        else:
            return OpenCVWindowKey.UNKNOWN