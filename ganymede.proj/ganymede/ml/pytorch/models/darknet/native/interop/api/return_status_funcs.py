# python
from ctypes import *
# project
from ..library_module import LIBRARY_MODULE
from ..data           import *


done_return_status = LIBRARY_MODULE.handler.done_return_status
done_return_status.argtypes = [
    return_status #return_status status
]
done_return_status.restype = None