# 3rd party
from copy import deepcopy
import cv2 as cv
# project
from ganymede.augmentation.parameters.mirror_parameters import MirrorParameters
from .parameters import *
from ganymede.augmentation.color import augmentation_basic_color_img
from .projection import augmentation_mirror, augmentation_padding, augmentation_rotate2d, augmentation_stretch


def augmentation_data_dynamic_params(data, params, inter_type = cv.INTER_AREA):
    p = params
    t = type(params)

    if t is BasicColorParameters:
        data = deepcopy(data)
        data.image = augmentation_basic_color_img(data.image, p.red, p.green, p.blue)
    elif t is MirrorParameters:
        data = augmentation_mirror(data, p.horizontal, p.vertical)
    elif t is PaddingParameters:
        data = augmentation_padding(data, p.left, p.right, p.top, p.bottom)
    elif t is Rotate3dParameters:
        raise NotImplementedError('not implement augmentation selection for rotate3d')
    elif t is Rotate2dParameters:
        data = augmentation_rotate2d(data, p.angle, inter_type)
    elif t is StretchParameters:
        data = augmentation_stretch(
            data,
            p.offset,
            p.type,
            p.orientation,
            inter_type
        )

    else:
        raise Exception(f'invalid augmentation params type:{t}')

    return data