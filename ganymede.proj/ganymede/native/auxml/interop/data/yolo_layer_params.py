# python
from ctypes import *
# project
from .c_list import *

class yolo_layer_params(Structure):
    _fields_ = [
        ('classes', c_int32),

        ('total', c_int32),

        ('mask', int_list),

        ('max_boxes', c_int32),

        ('scale_x_y', c_float),

        ('new_coords', c_int32), # bool original

        ('iou_normalizer', c_float),

        ('obj_normalizer', c_float),

        ('cls_normalizer', c_float),

        ('delta_normalizer', c_float),

        ('iou_loss', c_int32), # IoULossType

        ('iou_thresh_kind', c_int32), # IoULossType

        ('beta_nms', c_float),

        ('nms_kind', c_int32), # NmsKindType

        ('jitter', c_float),

        ('resize', c_float),

        ('ignore_thresh', c_float),

        ('truth_thresh', c_float),

        ('iou_thresh', c_float),
        
        ('random', c_float),

        ('anchors', anchor_list),
    ]