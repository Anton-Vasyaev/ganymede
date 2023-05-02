# python
from ctypes import *
# project
from ..auxml_library_module import LIBRARY_MODULE
from ..data           import *

api_process_yolo_sealed_output_detections = LIBRARY_MODULE.process_yolo_sealed_output_detections

api_process_yolo_sealed_output_detections.argtypes = [
    POINTER(float),         # float*          darknet_output,
    POINTER(c_int64),       # int64_t         boxes_count,
    c_int64,                # int64_t         batch_size,
    c_int32,                # int32_t         classes_count,
    c_float,                # float*          object_thresholds,
    POINTER(float),         # float*          nms_thresholds,
    POINTER(object_handler) # object_handler* detections_batch_handler // RETURN

]
api_process_yolo_sealed_output_detections.restype = return_status