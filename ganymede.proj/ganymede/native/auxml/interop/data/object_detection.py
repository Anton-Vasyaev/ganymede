# python
from ctypes import *

class object_detection(Structure):
    _fields_ = [
        ('x1', c_float),

        ('y1', c_float),

        ('x2', c_float),

        ('y2', c_float),

        ('class_id', c_int32),

        ('object_confidence', c_float),
        
        ('class_confidence', c_float),
    ]