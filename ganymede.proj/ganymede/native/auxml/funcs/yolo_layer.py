# python
from typing import List, Tuple
# 3rd party
import numpy as np
from ganymede.ml.data import ObjectDetection
# project
from ..interop import *
from ....ml.data import ObjectDetectionBatch, ObjectDetectionList
from ....ml.darknet.data import NetParams, YoloLayer

from ..auxiliary import *


def convert_int_list_to_native(values : List[int]) -> int_list:
    list_data = int_list()
    list_data.data = (c_int32 * len(values))()
    for value_idx in range(len(values)):
        list_data.data[value_idx] = int(values[value_idx])

    list_data.size = c_uint64(len(values))

    return list_data

def convert_anchor_list_to_native(values : List[Tuple[int, int]]) -> anchor_list:
    list_data = anchor_list()
    list_data.data = (anchor * len(values))()
    for value_idx in range(len(values)):
        x, y = values[value_idx]
        anchor_data = anchor(x, y)
        list_data.data[value_idx] = anchor_data

    list_data.size = c_uint64(len(values))

    return list_data
    

def convert_yolo_params_to_native(params : YoloLayer) -> yolo_layer_params:
    param_c = yolo_layer_params()

    param_c.classes = params.classes

    param_c.total = params.total

    param_c.mask = convert_int_list_to_native(params.mask)

    param_c.max_boxes = params.max_boxes

    param_c.scale_x_y = params.scale_x_y

    param_c.new_coords = params.new_coords

    param_c.iou_normalizer   = params.iou_normalizer
    param_c.obj_normalizer   = params.obj_normalizer
    param_c.cls_normalizer   = params.cls_normalizer
    param_c.delta_normalizer = params.delta_normalizer 

    param_c.iou_loss = int(params.iou_loss)
    param_c.iou_thresh_kind = int(params.iou_thresh_kind)
    param_c.beta_nms = params.beta_nms
    param_c.nms_kind = int(params.nms_kind)

    param_c.jitter = params.jitter
    param_c.resize = params.resize

    param_c.ignore_thresh = params.ignore_thresh
    param_c.truth_thresh  = params.truth_thresh
    param_c.iou_thresh    = params.iou_thresh

    param_c.random = params.random

    param_c.anchors = convert_anchor_list_to_native(params.anchors)

    return param_c


def convert_np_array_to_native(array : np.ndarray) -> output_tensor:
    tensor = output_tensor()
    
    b, h, w, c = array.shape

    tensor.batch    = b
    tensor.height   = h
    tensor.width    = w
    tensor.channels = c
    tensor.data     = cast(array.__array_interface__['data'][0], POINTER(c_float))

    return tensor 



def convert_yolo_modules_to_native(
    yolo_params  : List[YoloLayer],
    outputs : List[np.ndarray]
) -> List[yolo_output]:
    yolo_outputs : List[yolo_output] = []

    for idx in range(len(yolo_params)):
        yolo_param = yolo_params[idx]
        param_c    = convert_yolo_params_to_native(yolo_param)

        tensor_c = convert_np_array_to_native(outputs[idx])

        yolo_outputs.append(yolo_output(tensor_c, param_c))

    return yolo_outputs


def process_yolo_detections(
    output_layers_params  : List[YoloLayer],
    outputs               : List[np.ndarray],
    net_params_a          : NetParams,
    obj_thresholds        : List[float],
    nms_thresholds        : List[float]
) -> ObjectDetectionBatch:
    
    detections_batch_handler        = object_handler()
    detections_batch_handler.object = c_void_p(0)

    try:
        net_param_c = net_params()
        net_param_c.width = net_params_a.width
        net_param_c.height = net_params_a.height

        yolo_outputs_list = convert_yolo_modules_to_native(output_layers_params, outputs)

        yolo_outputs_array = (yolo_output * len(yolo_outputs_list))(*yolo_outputs_list)

        obj_thresholds_array = (c_float * len(obj_thresholds))(*obj_thresholds)
        nms_thresholds_array = (c_float * len(nms_thresholds))(*nms_thresholds)

        return_status = api_process_yolo_detections(
            yolo_outputs_array,
            len(yolo_outputs_array),
            net_param_c,
            obj_thresholds_array,
            nms_thresholds_array,
            detections_batch_handler
        )
        validate_return_status(return_status)

        return get_detections_from_handler(detections_batch_handler)
    finally:
        if detections_batch_handler.object != 0:
            api_object_handler_destroy(detections_batch_handler)