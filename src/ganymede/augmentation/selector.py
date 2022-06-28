# 3rd party
import cv2 as cv
# project
from ganymede.augmentation.parameters.mirror_parameters import MirrorParameters
from .parameters import *

from .basic_color import augmentation_basic_color
from .mirror      import augmentation_mirror
from .padding     import augmentation_padding
from .rotate3d    import augmentation_rotate3d
from .stretch     import augmentation_stretch


def augmentation_data_dynamic_params(data, params, inter_type = cv.INTER_AREA):
    p = params
    t = type(params)

    if t is BasicColorParameters:
        data = augmentation_basic_color(data, p.red, p.green, p.blue)
    elif t is MirrorParameters:
        data = augmentation_mirror(data, p.horizontal, p.vertical)
    elif t is PaddingParameters:
        data = augmentation_padding(data, p.left, p.right, p.top, p.bottom)
    elif t is Rotate3dParameters:
        data = augmentation_rotate3d(data, (p.x_angle, p.y_angle, p.z_angle), inter_type)
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