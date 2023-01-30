# python
from ctypes import *
# project
from ..library_module import LIBRARY_MODULE
from ..data           import *


process_yolo_detections = LIBRARY_MODULE.handler.process_yolo_detections
process_yolo_detections.argtypes = [
    POINTER(yolo_output),   # yolo_output* output_layers, // output type detections_batch_type
    c_uint64,               # uint64_t outputs_count,
    net_params,             # net_params params,
    c_float,                # float object_threshold,
    c_float,                # float nms_threshold,
    POINTER(object_handler) # object_handler* detections_batch_handler // RETURN
]
process_yolo_detections.restype = return_status