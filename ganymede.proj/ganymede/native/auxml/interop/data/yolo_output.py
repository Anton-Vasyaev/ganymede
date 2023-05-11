# python
from ctypes import *
# project
from .output_tensor import *
from .yolo_layer_params import *

class yolo_output(Structure):
    _fields_ = [
        ('output', output_tensor),

        ('params', yolo_layer_params),
    ]