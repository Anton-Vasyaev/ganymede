# python
from ctypes import *

class anchor(Structure):
    _fields_ = [
        ('x', c_float),

        ('y', c_float)
    ]