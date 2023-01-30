# python
from ctypes import *

class net_params(Structure):
    _fields_ = [
        ('width', c_int32),
        
        ('height', c_int32),
    ]