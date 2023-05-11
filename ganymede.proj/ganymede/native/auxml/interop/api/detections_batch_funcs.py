# python
from ctypes import *
# project
from ..auxml_library_module import LIBRARY_MODULE
from ..data import *


api_detections_batch_size = LIBRARY_MODULE.handler.detections_batch_size
api_detections_batch_size.argtypes = [
    object_handler,   # object_handler detections_batch_handler
    POINTER(c_uint64) # uint64_t* batch_size // RETURN
]
api_detections_batch_size.restype = return_status


api_detections_batch_detections_count = LIBRARY_MODULE.handler.detections_batch_detections_count
api_detections_batch_detections_count.argtypes = [
    object_handler,     # object_handler, detections_batch_handler,
    c_uint64,           # uint64_t batch_idx,
    POINTER(c_uint64),  # uint64_t* detections_count // RETURN
]
api_detections_batch_detections_count.restype = return_status


api_detections_batch_detections_store = LIBRARY_MODULE.handler.detections_batch_detections_store
api_detections_batch_detections_store.argtypes = [
    object_handler,             # object_handler detections_batch_handler,
    c_uint64,                   # uint64_t batch_idx,
    POINTER(object_detection)   # object_detection* detections
]
api_detections_batch_detections_store.restype = return_status
