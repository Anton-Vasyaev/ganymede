# python
from typing import List, Tuple
# 3rd party
import numpy as np
from ganymede.ml.data import ObjectDetection
# project
from .interop import *
from ..postprocessing.yolo_layer import ObjectDetectionBatch, ObjectDetectionList
from ..model.modules import YoloModule
from ..parsing.data import NetParams, YoloLayer


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


def validate_return_status(status : return_status):
    err_msg_s = ''

    if status.correct_status != 1:
        msg_len = LIBRARY_C.strlen(status.error_message)
        char_arr = (c_char * (msg_len + 1))()
        char_arr[-1] = 0
        memmove(byref(char_arr), status.error_message, msg_len)

        err_msg_s = bytearray(char_arr).decode('ascii')

    done_return_status(status)
    
    if err_msg_s != '':
        raise Exception(f'darknet_postprocessing native exception:{err_msg_s}')


def native_postprocessing_yolo_layer(
    output_modules : List[YoloModule],
    net_params_a   : NetParams,
    obj_thresh     : float,
    nms_thresh     : float
) -> ObjectDetectionBatch:
    
    detections_batch_handler = object_handler()
    detections_batch_handler.object = c_void_p(0)

    try:
        net_param_c = net_params()
        net_param_c.width = net_params_a.width
        net_param_c.height = net_params_a.height

        outputs : List[np.ndarray] = []
        for output_module in output_modules:
            output = output_module.output.detach().cpu().permute(0, 2, 3, 1).numpy()
            outputs.append(output)

        params = [m.params for m in output_modules]

        yolo_outputs_list = convert_yolo_modules_to_native(params, outputs)

        yolo_outputs_array = (yolo_output * len(yolo_outputs_list))(*yolo_outputs_list)

        return_status = process_yolo_detections(
            yolo_outputs_array,
            len(yolo_outputs_array),
            net_param_c,
            obj_thresh,
            nms_thresh,
            detections_batch_handler
        )
        validate_return_status(return_status)

        batch_size_c = c_uint64()
        return_status = detections_batch_size(
            detections_batch_handler,
            pointer(batch_size_c)
        )
        validate_return_status(return_status)
        batch_size = batch_size_c.value

        detections_batch : ObjectDetectionBatch = []

        for batch_idx in range(batch_size):
            batch_idx_c = c_uint64(batch_idx)

            detections_count_c = c_uint64()
            return_status = detections_batch_detections_count(
                detections_batch_handler,
                batch_idx_c,
                byref(detections_count_c)
            )
            validate_return_status(return_status)

            detections_count = detections_count_c.value
            object_detections_arr = (object_detection * detections_count)()
            return_status = detections_batch_detections_store(
                detections_batch_handler,
                batch_idx_c,
                object_detections_arr
            )

            detections : ObjectDetectionList = []

            for obj_det in object_detections_arr:
                x1 : float = obj_det.x1
                y1 : float = obj_det.y1
                x2 : float = obj_det.x2
                y2 : float = obj_det.y2

                class_id : int = obj_det.class_id

                obj_conf : float = obj_det.object_confidence
                cls_conf : float = obj_det.class_confidence

                detections.append(ObjectDetection((x1, y1, x2, y2), class_id, obj_conf, cls_conf))

            detections_batch.append(detections)

        return detections_batch
    finally:
        if detections_batch_handler.object != 0:
            object_handler_destroy(detections_batch_handler)