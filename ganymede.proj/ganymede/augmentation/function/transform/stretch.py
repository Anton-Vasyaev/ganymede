# python
from typing import Tuple
# 3rd party
import cv2 as cv # type: ignore

# project
from ganymede.augmentation.augmentation_data import AugmentationData
from ganymede.augmentation.function.transform.stretch_data import StretchType, StretchOrientation
from ganymede.augmentation.function.transform.perspective import augmentate_perspective

from ganymede.opencv import CvPerspective


def __validate_offset(
    offset: float,
    img_size: Tuple[int, int],
    orientation: StretchOrientation
):
    img_w, img_h = img_size

    if offset <= -1.0 or offset >= 1.0:
        raise Exception(f'invalid offset value:{offset}')

    dim = img_w if orientation == StretchOrientation.HORIZONTAL else img_h

    start = 0.0
    end = 0.0
    if offset < 0.0:
        start = 0
        end = dim - dim * offset
    else:
        start = dim * offset
        end = dim

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
    img_size: Tuple[int, int],
    orientation
) -> CvPerspective:
    img_w, img_h = img_size

    lt_p = 0.0, 0.0
    rt_p = 0.0, 0.0
    lb_p = 0.0, 0.0
    rb_p = 0.0, 0.0

    if orientation == StretchOrientation.HORIZONTAL:
        if offset > 0:
            lt_p = offset, 0.0
            rt_p = 1.0, 0.0
            lb_p = 0.0, 1.0
            rb_p = 1.0 - offset, 1.0
        else:
            lt_p = 0.0, 0.0
            rt_p = 1.0 - abs(offset), 0.0
            lb_p = abs(offset), 1.0
            rb_p = 1.0, 1.0
    else:
        if offset > 0:
            lt_p = 0.0, offset
            rt_p = 1.0, 0.0
            lb_p = 0.0, 1.0
            rb_p = 1.0, 1.0 - offset
        else:
            lt_p = 0.0, 0.0
            rt_p = 1.0, abs(offset)
            lb_p = 0.0, 1.0 - abs(offset)
            rb_p = 1.0, 1.0

    return (lt_p, rt_p, lb_p, rb_p)


def augmentate_stretch(
    data: AugmentationData,
    offset: float,
    orientation: StretchOrientation = StretchOrientation.HORIZONTAL,
    type: StretchType = StretchType.SRC,
    interpolation: int = cv.INTER_AREA
) -> AugmentationData:
    img = data.image
    img_h, img_w = img.shape[0:2]

    __validate_offset(offset, (img_w, img_h), orientation)

    src_points = __get_stretch_points(offset, (img_w, img_h), orientation)

    dst_points = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))

    if type == StretchType.DST:
        src_points, dst_points = dst_points, src_points

    return augmentate_perspective(data, src_points, dst_points, interpolation)
