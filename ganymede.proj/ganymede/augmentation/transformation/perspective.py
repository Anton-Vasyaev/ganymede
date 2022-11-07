# 3rd party
import cv2 as cv
# project
import ganymede.opencv as g_cv
from ganymede.augmentation.augmentation_data import AugmentationData
from ganymede.augmentation.transformation.perspective_coord_transformer import PerspectiveCoordTransformer
from ganymede.augmentation.transformation.delegate import delegate_transform_points

from ganymede.opencv import PerspectiveType


def augmentate_perspective(
    data: AugmentationData,
    src_points: PerspectiveType,
    dst_points: PerspectiveType,
    interpolation=cv.INTER_AREA
) -> AugmentationData:
    img = g_cv.warp_perspective_on_keypoints(
        data.image, src_points, dst_points, interpolation
    )

    transform = PerspectiveCoordTransformer.from_points(src_points, dst_points)

    points = delegate_transform_points(data.points, transform)

    return AugmentationData(img, points)
