# python
from ctypes import *

class output_tensor(Structure):
    _fields_ = [
        ('data', POINTER(c_float)), 

        ('batch', c_uint64),

        ('width', c_uint64),

        ('height', c_uint64),

        ('channels', c_uint64),
    ]