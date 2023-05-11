# python
import math
from typing import List, Tuple
# 3rd party
import torch
import numpy as np
import ganymede.math.bbox2 as m_bbox2
from ganymede.ml.data import *
from nameof import nameof
# project
from ..model.modules        import YoloModule
from ..data                 import YoloBox
from .....darknet.parse_blocksckscks import NetParams


YoloDetectionList  = List[YoloBox]
YoloDetectionBatch = List[YoloDetectionList]



def sigmoid(x):
    return 1 / (1 + math.exp(-x))

sigmoid_v = np.vectorize(sigmoid)


def scale_x_y(data, scale_x_y):
    return data * scale_x_y - 0.5 * (scale_x_y - 1.0)

scale_x_y_v = np.vectorize(scale_x_y)


def sigmoid_scale_x_y(data, scale_x_y):
    return sigmoid(data) * scale_x_y - 0.5 * (scale_x_y - 1.0)

sigmoid_scale_x_y_v = np.vectorize(sigmoid_scale_x_y)

'''
def activate_yolo_output(
    batch_yolo_output : np.ndarray,
    anchors_count     : int,
    classes_count     : int,
    scale_x_y         : float
) -> np.ndarray:
    # x, y, w, h, 
    mask_size    = (5 + classes_count)
    filters_size = anchors_count * mask_size

    out_b, out_h, out_w, out_c = batch_yolo_output.shape

    yolo_output_data = batch_yolo_output.flatten().tolist()

    for idx in range(len(yolo_output_data)):
        remain_idx = idx % mask_size
        if remain_idx >= 0 and remain_idx <= 1:
            yolo_output_data[idx] = sigmoid(yolo_output_data[idx]) * scale_x_y - 0.5 * (scale_x_y - 1.0)
        elif remain_idx >= 4 and remain_idx < mask_size:
            yolo_output_data[idx] = sigmoid(yolo_output_data[idx])

    activate_output = np.array(yolo_output_data)
    activate_output.shape = batch_yolo_output.shape

    return activate_output
'''

def activate_yolo_output(
    batch_yolo_output : np.ndarray,
    anchors_count     : int,
    classes_count     : int,
    scale_x_y         : float,
    new_coords        : bool = True
) -> np.ndarray:
    mask_size = (5 + classes_count)
    filters_size = anchors_count * mask_size

    output = batch_yolo_output.copy()

    # convert shape from batch, h, w, filters -> batch, h, w, anchors_count, mask_size
    original_shape = output.shape
    output.shape   = output.shape[:-1] + (anchors_count, mask_size)

    if not new_coords:
        # activate sigmoid  x, y
        output[...,0:2] = sigmoid_v(output[..., 0:2])

        # activate sigmoid object prob and classes probs
        output[...,4:] = sigmoid_v(output[..., 4:])

    # scale x, y
    output[..., 0:2] = scale_x_y_v(output[..., 0:2], scale_x_y)

    output.shape = original_shape

    return output


def get_yolo_boxes(
    activated_output   : np.ndarray,
    anchors            : List[Tuple[int, int]],
    classes_count      : int,
    net_width          : int,
    net_height         : int,
    new_coords         : bool,
    obj_conf_threshold : float
) -> YoloDetectionBatch:
    mask_size    = (5 + classes_count)
    filters_size = len(anchors) * mask_size

    data = activated_output.tolist()

    out_b, out_h, out_w, out_c = activated_output.shape

    detections_batch : YoloDetectionBatch = list()

    for b_i in range(out_b):
        detections : YoloDetectionList = list()

        for h_i in range(out_h):
            for w_i in range(out_w):
                for anchor_i in range(len(anchors)):
                    curr_anchor = anchors[anchor_i]
                    w_anchor, h_anchor = curr_anchor

                    anchor_start = anchor_i * mask_size
                    anchor_end   = anchor_start + mask_size
                    box_values : List[float] = data[b_i][h_i][w_i][anchor_start:anchor_end]

                    x, y, w, h, obj_conf = box_values[:5]

                    # objc_conf < obj_conf_threshold incorrect behavior for NaN values
                    if not obj_conf > obj_conf_threshold:
                        continue

                    classes_probs = box_values[5:]

                    for class_idx in range(len(classes_probs)):
                        class_prob = classes_probs[class_idx]
                        prob = obj_conf * class_prob
                        classes_probs[class_idx] = prob if prob > obj_conf_threshold else 0.0

                    x = (w_i + x) / out_w
                    y = (h_i + y) / out_h

                    if new_coords:
                        w = w * w * 4 * w_anchor / net_width
                        h = h * h * 4 * h_anchor / net_height
                    else:
                        w = math.exp(w) * w_anchor / net_width
                        h = math.exp(h) * h_anchor / net_height

                    x1, y1 = x - w / 2, y - h / 2
                    x2, y2 = x + w / 2, y + h / 2

                    detections.append(YoloBox((x1, y1, x2, y2), obj_conf, classes_probs))

        detections_batch.append(detections)

    return detections_batch
    

def do_nms_sort(
    detections_batch : YoloDetectionBatch,
    classes_count    : int,
    nms_thresh       : float
) -> None:
    for detections in detections_batch:
        for class_idx in range(classes_count):
            detections.sort(key = lambda b : b.class_probs[class_idx], reverse=True)

            for i in range(len(detections)):
                main_box  = detections[i]
                main_prob = main_box.class_probs[class_idx] 

                if main_prob == 0.0:
                    continue

                for j in range(i + 1, len(detections)):
                    curr_box  = detections[j]
                    curr_prob = curr_box.class_probs[class_idx] 
                    
                    if curr_prob == 0.0:
                        continue

                    if m_bbox2.iou(main_box.bbox, curr_box.bbox) >= nms_thresh:
                        curr_box.class_probs[class_idx] = 0.0


def postprocessing_yolo_layer(
    output_modules : List[YoloModule],
    net_params     : NetParams,
    obj_thresh     : float,
    nms_thresh     : float
) -> ObjectDetectionBatch:
    yolo_detections_batch : YoloDetectionBatch = list()

    for idx in range(len(output_modules)):
        module = output_modules[idx]
        params = module.params

        anchors = params.anchors
        mask    = params.mask

        curr_anchors : List[Tuple[int, int]] = list() 
        for mask_idx in mask:
            curr_anchors.append(anchors[mask_idx])

        output_np : np.ndarray = module.output.detach().cpu().numpy()
        output_np = output_np.transpose(0, 2, 3, 1)

        # activate outputs
        activated_output = activate_yolo_output(output_np, len(curr_anchors), params.classes, params.scale_x_y, params.new_coords)

        # build box candidates
        yolo_boxes_batch = get_yolo_boxes(
            activated_output,
            curr_anchors,
            params.classes,
            net_params.width,
            net_params.height,
            params.new_coords,
            obj_thresh
        )

        if len(yolo_detections_batch) == 0:
            yolo_detections_batch = yolo_boxes_batch
        else:
            for idx in range(len(yolo_boxes_batch)):
                yolo_detections_batch[idx] += yolo_boxes_batch[idx]

    # process Non Maximum Supression
    do_nms_sort(yolo_detections_batch, params.classes, nms_thresh)

    
    # make object detections
    detections_batch : ObjectDetectionBatch = list()

    for yolo_detections in yolo_detections_batch:
        detections : ObjectDetectionList = list()
        for yolo_detection in yolo_detections:
            bbox        = yolo_detection.bbox
            obj_conf     = yolo_detection.obj_conf
            class_probs = yolo_detection.class_probs

            class_idx      = int(np.argmax(class_probs))
            max_class_prob = class_probs[class_idx]

            if max_class_prob != 0.0:
                detections.append(
                    ObjectDetection(
                        bbox, 
                        class_idx, 
                        obj_conf, 
                        max_class_prob
                    )
                )

        detections_batch.append(detections)

    return detections_batch