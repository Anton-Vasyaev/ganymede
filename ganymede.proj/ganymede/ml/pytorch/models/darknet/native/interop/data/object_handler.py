# python
from ctypes import *


class object_handler(Structure):
    _fields_ = [
        ('object', c_void_p),   # void*
        
        ('type', c_int32)       # object_type
    ]