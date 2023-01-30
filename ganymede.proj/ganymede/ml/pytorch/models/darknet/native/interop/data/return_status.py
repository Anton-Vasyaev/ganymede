# python
from ctypes import *

class return_status(Structure):
    _fields_ = [
        ('correct_status', c_int32),

        ('error_message', POINTER(c_char)), 

        ('is_const_message', c_int32),
    ]
