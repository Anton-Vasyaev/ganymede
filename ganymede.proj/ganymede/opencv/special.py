# python
from typing import Tuple, cast
# 3rd party
import cv2 as cv # type: ignore
import numpy as np
# project
from ganymede.math.primitives import Mat3x3, Point2

PerspectiveType = Tuple[Point2, Point2, Point2, Point2]


def canny_channels(
    img: np.ndarray,
    threshold_1 : int,
    threshold_2 : int
) -> np.ndarray:
    original_type = img.dtype
    if np.issubdtype(original_type, np.floating):
        img = (img * 255).astype(np.uint8)

    canny_r = cv.Canny(img[..., 0], threshold_1,
                       threshold_2).astype(np.float32)
    canny_g = cv.Canny(img[..., 1], threshold_1,
                       threshold_2).astype(np.float32)
    canny_b = cv.Canny(img[..., 2], threshold_1,
                       threshold_2).astype(np.float32)

    canny_sum = canny_r + canny_g + canny_b
    canny_sum[canny_sum > 255] = 255

    canny_sum = canny_sum.astype(np.uint8)

    return canny_sum


def equalize_hist_color_yuv(bgr_img: np.ndarray) -> np.ndarray:
    yuv = cv.cvtColor(bgr_img, cv.COLOR_BGR2YUV)

    yuv[..., 0] = cv.equalizeHist(yuv[..., 0])

    bgr_img = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)

    return bgr_img


def equalize_hist_color_channel(bgr_img: np.ndarray) -> np.ndarray:
    eq_img = bgr_img.copy()

    for c_i in range(3):
        eq_img[..., c_i] = cv.equalizeHist(eq_img[..., c_i])

    return eq_img


def colors_sum_norm(img: np.ndarray) -> np.ndarray:
    original_type = img.dtype
    if not np.issubdtype(original_type, np.floating):
        img = img.astype(np.float32) / 255

    sum = img[..., 0] + img[..., 1] + img[..., 2]

    max_val = sum.max()
    min_val = sum.min()

    distance = max_val - min_val

    sum -= min_val
    sum /= distance

    if not np.issubdtype(original_type, np.floating):
        sum = (sum * 255).astype(np.uint8)

    return sum


def get_perspective_transform(src_points: PerspectiveType, dst_points: PerspectiveType) -> Mat3x3:
    src = cast(np.floating, src_points)
    dst = cast(np.floating, dst_points)
    src_p = np.float32(src)
    dst_p = np.float32(dst)

    mat_data = cast(np.ndarray, cv.getPerspectiveTransform(src_p, dst_p))

    mat = mat_data.tolist()

    return (
        (mat[0][0], mat[0][1], mat[0][2]),
        (mat[1][0], mat[1][1], mat[1][2]),
        (mat[2][0], mat[2][1], mat[2][2]),
    )
