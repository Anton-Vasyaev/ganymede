# 3rd party
import cv2 as cv
import numpy as np

# project
from ganymede.augmentation.augmentation_data import AugmentationData
from ganymede.augmentation.parameters        import StretchType, StretchOrientation
from .perspective                            import augmentation_perspective


def __validate_offset(
    offset,
    img_size,
    orientation
):
    img_w, img_h = img_size

    if offset <= -1.0 or offset >= 1.0:
        raise Exception(f'invalid offset value:{offset}')

    dim = img_w if orientation == StretchOrientation.HORIZONTAL else img_h

    start = None
    end   = None
    if offset < 0.0:
        start = 0
        end   = dim - dim * offset
    else:
        start = dim * offset
        end   = dim

    if start >= end:
        raise Exception(
            f'offset collision problem;'
            f'start >= end;'
            f'{start} >= {end};'
            f'offset:{offset};'
            f'orientation:{orientation}'
        )


def __get_stretch_points(
    offset,
    img_size,
    orientation
):
    img_w, img_h = img_size

    lt_p = None
    rt_p = None
    lb_p = None
    rb_p = None

    if orientation == StretchOrientation.HORIZONTAL:
        if offset > 0:
            lt_p = (int(img_w * abs(offset)), 0)
            rt_p = (img_w, 0)
            lb_p = (0, img_h)
            rb_p = (int(img_w - img_w * abs(offset)), img_h)
        else:
            lt_p = (0, 0)
            rt_p = (int(img_w - img_w * abs(offset)), 0)
            lb_p = (int(img_w * abs(offset)), img_h)
            rb_p = (img_w, img_h)
    else:
        if offset > 0:
            lt_p = (0, img_h * abs(offset))
            rt_p = (img_w, 0)
            lb_p = (0, img_h)
            rb_p = (img_w, int(img_h - img_h * abs(offset)))
        else:
            lt_p = (0, 0)
            rt_p = (img_w, int(img_h * abs(offset)))
            lb_p = (0, int(img_h - img_h * abs(offset)))
            rb_p = (img_w, img_h)

    return [lt_p, rt_p, lb_p, rb_p]


def augmentation_stretch(
    data           : AugmentationData,
    offset         : float,
    type           : StretchType        = StretchType.SRC,
    orientation    : StretchOrientation = StretchOrientation.HORIZONTAL,
    interpolation  : int                = cv.INTER_AREA
) -> AugmentationData:
    new_data = {}

    img = data['image']
    img_h, img_w = img.shape[0:2]

    __validate_offset(offset, (img_w, img_h), orientation)

    src_points = __get_stretch_points(offset, (img_w, img_h), orientation)

    dst_points = [[0, 0], [img_w, 0], [0, img_h], [img_w, img_h]]

    if type == StretchType.DST: src_points, dst_points = dst_points, src_points
    
    return augmentation_perspective(data, src_points, dst_points, interpolation)