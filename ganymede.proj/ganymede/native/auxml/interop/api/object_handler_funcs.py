# python
from ctypes import *
# project
from ..auxml_library_module import LIBRARY_MODULE
from ..data import *



api_object_handler_destroy = LIBRARY_MODULE.handler.object_handler_destroy
api_object_handler_destroy.argtypes = [
    object_handler
]
api_object_handler_destroy.restype = None