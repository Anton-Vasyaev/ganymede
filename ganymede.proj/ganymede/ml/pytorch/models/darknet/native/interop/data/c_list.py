# python
from ctypes import *
# project
from .anchor import anchor

class int_list(Structure):
    _fields_ = [
        ('data', POINTER(c_int32)),

        ('size', c_uint64)
    ]


class float_list(Structure):
    _fields_ = [
        ('data', POINTER(c_float)), 

        ('size', c_uint64)
    ]


class anchor_list(Structure):
    _fields_ = [
        ('data', POINTER(anchor)),

        ('size', c_uint64)
    ]