# python
from typing import Optional

# 3rd party
import cv2 as cv

# project
from ganymede.imaging import ImageType


CV_COLOR_CONVERSIONS_CODE = {
    ImageType.RGB: {
        ImageType.BGR: cv.COLOR_RGB2BGR,

        ImageType.GRAY: cv.COLOR_RGB2GRAY,

        ImageType.RGBA: cv.COLOR_RGB2RGBA,

        ImageType.BGRA: cv.COLOR_RGB2BGRA,
    },
    ImageType.GRAY: {
        ImageType.RGB: cv.COLOR_GRAY2RGB,

        ImageType.BGR: cv.COLOR_GRAY2BGR,

        ImageType.RGBA: cv.COLOR_GRAY2RGBA,

        ImageType.BGRA: cv.COLOR_GRAY2BGRA,
    },
    ImageType.RGBA: {
        ImageType.RGB: cv.COLOR_RGBA2RGB,

        ImageType.BGR: cv.COLOR_RGBA2BGR,

        ImageType.GRAY: cv.COLOR_RGBA2GRAY,

        ImageType.BGRA: cv.COLOR_RGBA2BGRA,
    },
    ImageType.BGRA: {
        ImageType.RGB: cv.COLOR_BGRA2RGB,

        ImageType.BGR: cv.COLOR_BGRA2BGR,

        ImageType.GRAY: cv.COLOR_BGRA2GRAY,
        
        ImageType.RGBA: cv.COLOR_BGRA2RGBA,
    },
}


def get_cv_color_conversion_code(
    src_type: ImageType, 
    dst_type: ImageType
) -> Optional[int]:
    if src_type == dst_type:
        return None

    try:
        code = CV_COLOR_CONVERSIONS_CODE[src_type][dst_type]

        return code
    except:
        raise ValueError(
            f"not find color conversions from '{src_type}' to '{dst_type}'."
        )
