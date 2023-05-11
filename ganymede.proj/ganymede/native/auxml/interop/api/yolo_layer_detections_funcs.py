# python
from ctypes import *
# project
from ..auxml_library_module import LIBRARY_MODULE
from ..data           import *


api_process_yolo_detections = LIBRARY_MODULE.handler.process_yolo_detections
api_process_yolo_detections.argtypes = [
    POINTER(yolo_output),   # yolo_output* output_layers, // output type detections_batch_type
    c_uint64,               # uint64_t outputs_count,
    net_params,             # net_params params,
    POINTER(c_float),       # float* object_thresholds,
    POINTER(c_float),       # float* nms_thresholds,
    POINTER(object_handler) # object_handler* detections_batch_handler // RETURN
]
api_process_yolo_detections.restype = return_status