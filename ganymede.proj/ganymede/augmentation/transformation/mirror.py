# 3rd party
import cv2 as cv
import numpy as np
# project
from ganymede.math.primitives import Point2
from ganymede.augmentation.augmentation_data import AugmentationData
from ganymede.augmentation.transformation.delegate import delegate_transform_points


class MirrorCoordTransformer:
    horizontal: bool
    vertical: bool

    def __init__(
        self,
        horizontal: bool,
        vertical: bool
    ):
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, coord: Point2) -> Point2:
        x, y = coord

        if self.horizontal:
            x = 1.0 - x
        if self.vertical:
            y = 1.0 - y

        return x, y


def __augmentate_img_mirror(
    img: np.ndarray,
    horizontal: bool,
    vertical: bool
):
    flip_value = None

    if not horizontal and not vertical:
        return img.copy()
    elif horizontal and vertical:
        flip_value = -1
    elif horizontal:
        flip_value = 1
    elif vertical:
        flip_value = 0

    img = cv.flip(img, flip_value)

    return img


def augmentate_mirror(
    data: AugmentationData,
    horizontal: bool = True,
    vertical: bool = True
) -> AugmentationData:

    image = __augmentate_img_mirror(data.image, horizontal, vertical)

    transformer = MirrorCoordTransformer(horizontal, vertical)

    points = delegate_transform_points(data.points, transformer)

    return AugmentationData(image, points)
