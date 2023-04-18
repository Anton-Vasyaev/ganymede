# python
from ctypes import *
# project
from ..auxml_library_module import LIBRARY_MODULE
from ..data           import *


api_done_return_status = LIBRARY_MODULE.handler.done_return_status
api_done_return_status.argtypes = [
    return_status #return_status status
]
api_done_return_status.restype = None