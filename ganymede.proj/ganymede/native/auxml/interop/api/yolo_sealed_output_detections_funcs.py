# python
from ctypes import *
# project
from ..auxml_library_module import LIBRARY_MODULE
from ..data           import *

api_process_yolo_sealed_output_detections = LIBRARY_MODULE.handler.process_yolo_sealed_output_detections

api_process_yolo_sealed_output_detections.argtypes = [
    POINTER(c_float),       # float*          darknet_output,
    c_int64,                # int64_t         boxes_count,
    c_int64,                # int64_t         batch_size,
    c_int32,                # int32_t         classes_count,
    POINTER(c_float),       # float*          object_thresholds,
    POINTER(c_float),       # float*          nms_thresholds,
    c_int32,                # int32_t         net_width,
    c_int32,                # int32_t         net_height,
    POINTER(object_handler) # object_handler* detections_batch_handler // RETURN
]
api_process_yolo_sealed_output_detections.restype = return_status