# python
from dataclasses import dataclass
from typing import List, Tuple
# project
from ..enum_types.iou_loss_type import IoULossType
from ..enum_types.nms_kind_type import NmsKindType
from ...option import *


@dataclass
class YoloLayer:
    classes : int

    total : int

    mask : List[int]

    max_boxes : int

    scale_x_y : float

    new_coords : bool

    iou_normalizer : float

    obj_normalizer : float

    cls_normalizer : float

    delta_normalizer : float

    iou_loss : IoULossType

    iou_thresh_kind : IoULossType

    beta_nms : float

    nms_kind : NmsKindType

    jitter : float

    resize : float

    ignore_thresh : float

    truth_thresh : float

    iou_thresh : float

    random : float

    anchors : List[Tuple[int, int]]

    config_block : ConfigBlock


ALLOW_YOLO_PARAMS = set([
    'classes',
    'num', # total
    'mask',
    'max', # max_boxes
    'scale_x_y',
    'new_coords',
    'iou_normalizer',
    'obj_normalizer',
    'cls_normalizer',
    'delta_normalizer',
    'iou_loss',
    'iou_thresh_kind',
    'beta_nms',
    'nms_kind',
    'jitter',
    'resize',
    'ignore_thresh',
    'truth_thresh',
    'iou_thresh',
    'random',
    'anchors'
])


def load_anchors_from_int_list(data_list : List[int]) -> List[Tuple[int, int]]:
    anchors : List[Tuple[int, int]] = []
    for idx in range(len(data_list) // 2):
        anchors.append((data_list[2*idx], data_list[2*idx+1]))

    return anchors


def parse_yolo(data : ConfigBlock) -> YoloLayer:
    option_validate_allow_parameters(data, ALLOW_YOLO_PARAMS)

    classes = option_find_int_default(data, 'classes', 20)

    total = option_find_int_default(data, 'total', 1)

    mask = option_find_int_list(data, 'mask')

    max_boxes = option_find_int_default(data, 'max', 200)

    scale_x_y = option_find_float_default(data, 'scale_x_y', 1.0)

    new_coords = option_find_int_default(data, 'new_coords', 0)

    iou_normalizer = option_find_float_default(data, 'iou_normalizer', 0.75)

    obj_normalizer = option_find_float_default(data, 'obj_normalizer', 1.0)

    cls_normalizer = option_find_float_default(data, 'cls_normalizer', 1.0)

    delta_normalizer = option_find_float_default(data, 'delta_normalizer', 1.0)

    iou_loss = IoULossType.from_str(option_find_str_default(data, 'iou_loss', 'mse'))

    iou_thresh_kind = IoULossType.from_str(option_find_str_default(data, 'iou_thresh_kind', 'iou'))

    beta_nms = option_find_float_default(data, 'beta_nms', 0.6)

    nms_kind = NmsKindType.from_str(option_find_str_default(data, 'nms_kind', 'default'))

    jitter = option_find_float_default(data, 'jitter', 0.2)

    resize = option_find_float_default(data, 'resize', 1.0)

    ignore_thresh = option_find_float_default(data, 'ignore_thresh', 0.5)
    
    truth_thresh = option_find_float_default(data, 'truth_thresh', 1.0)

    iou_thresh = option_find_float_default(data, 'iou_thresh', 1.0)

    random = option_find_float_default(data, 'random', 0)

    anchors = load_anchors_from_int_list(option_find_int_list(data, 'anchors'))

    return YoloLayer(
        classes,
        total,
        mask,
        max_boxes,
        scale_x_y,
        bool(new_coords),
        iou_normalizer,
        obj_normalizer,
        cls_normalizer,
        delta_normalizer,
        iou_loss,
        iou_thresh_kind,
        beta_nms,
        nms_kind,
        jitter,
        resize,
        ignore_thresh,
        truth_thresh,
        iou_thresh,
        random,
        anchors,
        data
    )