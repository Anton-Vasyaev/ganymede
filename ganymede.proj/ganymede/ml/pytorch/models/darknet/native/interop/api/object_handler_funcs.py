# python
from ctypes import *
# project
from ..library_module import LIBRARY_MODULE
from ..data import *



object_handler_destroy = LIBRARY_MODULE.handler.object_handler_destroy
object_handler_destroy.argtypes = [
    object_handler
]
object_handler_destroy.restype = None