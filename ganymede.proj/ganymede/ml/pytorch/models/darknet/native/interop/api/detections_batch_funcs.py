# python
from ctypes import *
# project
from ..library_module import LIBRARY_MODULE
from ..data import *


detections_batch_size = LIBRARY_MODULE.handler.detections_batch_size
detections_batch_size.argtypes = [
    object_handler,   # object_handler detections_batch_handler
    POINTER(c_uint64) # uint64_t* batch_size // RETURN
]
detections_batch_size.restype = return_status


detections_batch_detections_count = LIBRARY_MODULE.handler.detections_batch_detections_count
detections_batch_detections_count.argtypes = [
    object_handler,     # object_handler, detections_batch_handler,
    c_uint64,           # uint64_t batch_idx,
    POINTER(c_uint64),  # uint64_t* detections_count // RETURN
]
detections_batch_detections_count.restype = return_status


detections_batch_detections_store = LIBRARY_MODULE.handler.detections_batch_detections_store
detections_batch_detections_store.argtypes = [
    object_handler,             # object_handler detections_batch_handler,
    c_uint64,                   # uint64_t batch_idx,
    POINTER(object_detection)   # object_detection* detections
]
detections_batch_detections_store.restype = return_status
