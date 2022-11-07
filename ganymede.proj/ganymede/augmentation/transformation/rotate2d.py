# python
from typing import cast
# 3rd party
import cv2 as cv
# project
import ganymede.opencv as g_cv
import ganymede.math.convert as gm_convert
import ganymede.math.point2 as gm_p2

from ganymede.opencv import CvPerspective

from ganymede.augmentation.augmentation_data import AugmentationData
from ganymede.augmentation.transformation.perspective_coord_transformer import PerspectiveCoordTransformer
from ganymede.augmentation.transformation.delegate import delegate_transform_points


def augmentate_rotate2d(
    data: AugmentationData,
    angle: float,
    interpolation: int = cv.INTER_AREA
) -> AugmentationData:
    img_h, img_w = data.image.shape[0:2]
    img_size_ratio = img_w / img_h

    rel_w = 1.0
    rel_h = 1.0 / img_size_ratio

    c_x = rel_w / 2
    c_y = rel_h / 2

    angle = -gm_convert.deg2rad(angle)

    src_points = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))

    prepare_points = [(0.0, 0.0), (rel_w, 0.0), (0.0, rel_h), (rel_w, rel_h)]

    rotate_points = [gm_p2.rotate(p, angle, (c_x, c_y))
                     for p in prepare_points]

    dst_points_list = gm_p2.normalize_on_self(rotate_points)

    dst_points = cast(CvPerspective, tuple(dst_points_list))

    l, t, r, b = gm_p2.get_contour(rotate_points)
    w_scale, h_scale = (r - l) / rel_w, (b - t) / rel_h
    output_size = int(w_scale * img_w), int(h_scale * img_h)

    img = g_cv.warp_perspective_on_keypoints(
        data.image, src_points, dst_points, interpolation, output_size)

    transform = PerspectiveCoordTransformer.from_points(
        src_points, dst_points)

    points = delegate_transform_points(data.points, transform)

    return AugmentationData(img, points)
