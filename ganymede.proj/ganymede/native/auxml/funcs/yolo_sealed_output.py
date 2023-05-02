# python
from ctypes import *
from typing import List
# 3rd party
import numpy as np

from nameof import nameof
# project
import ganymede.numpy.validation as np_valid
from ganymede.ml.data.object_detection import ObjectDetection, ObjectDetectionBatch, ObjectDetectionList

from ..interop import *
from ..auxiliary import validate_return_status, get_detections_from_handler

def __validate_yolo_sealaed_output_array(
    yolo_output       : np.ndarray,
    object_thresholds : List[float],
    nms_thresholds    : List[float]
):
    np_valid.number_of_dimensions_equal(yolo_output, 3, nameof(yolo_output))
    np_valid.is_float32(yolo_output, nameof(yolo_output))

    batch_size, _, _ = yolo_output.shape

    if len(object_thresholds) != len(nms_thresholds):
        raise Exception(
            f'len of object thresholds != len of nms thresholds:'
            f'{len(object_thresholds)} != {len(nms_thresholds)}.'
        )
    
    if batch_size != len(object_thresholds):
        raise Exception(
            f'batch size and numbers of object/nms threshold are not equal:'
            f'{batch_size} != {len(object_thresholds)}.'
        )
    

def process_yolo_sealead_output_detections(
    yolo_output : np.ndarray,
    object_thresholds : List[float],
    nms_thresholds    : List[float]
) -> ObjectDetectionBatch:
    __validate_yolo_sealaed_output_array(yolo_output, object_thresholds, nms_thresholds)

    batch_size, boxes_count, values_count = yolo_output.shape
    classes_count = values_count - 5


    data_ptr_c = cast(yolo_output.__array_interface__['data'][0], POINTER(c_float))
    boxes_count_c = c_int64(boxes_count)
    batch_size_c  = c_int64(batch_size)
    classes_count_c = c_int32(classes_count)

    obj_thresholds_array = (c_float * len(object_thresholds))(*object_thresholds)
    nms_thresholds_array = (c_float * len(nms_thresholds))(*nms_thresholds)

    detections_batch_handler        = object_handler()
    detections_batch_handler.object = c_void_p(0)

    try:
        return_status = api_process_yolo_sealed_output_detections(
            data_ptr_c,
            boxes_count_c,
            batch_size_c,
            classes_count_c,
            obj_thresholds_array,
            nms_thresholds_array,
            detections_batch_handler
        )
        validate_return_status(return_status)

        return get_detections_from_handler(detections_batch_handler)
    finally:
        if detections_batch_handler.object != 0:
            api_object_handler_destroy(detections_batch_handler)