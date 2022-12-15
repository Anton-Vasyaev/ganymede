# python
from typing import Tuple, cast
# 3rd party
import cv2 as cv # type: ignore
import numpy as np
# project
from ganymede.opencv.data import CvPerspective


def resize_frame(
    img: np.ndarray,
    frame_size: Tuple[int, int],
    interpolation: int = cv.INTER_AREA
) -> np.ndarray:
    frame_w, frame_h = frame_size

    img_h, img_w = img.shape[0:2]

    width_scale, height_scale = frame_w / img_w, frame_h / img_h

    scale = min(width_scale, height_scale)

    resize_w, resize_h = int(img_w * scale), int(img_h * scale)

    img = cv.resize(img, (resize_w, resize_h), interpolation=interpolation)

    return img


def warp_perspective_on_keypoints(
    img: np.ndarray,
    src_points: CvPerspective,
    dst_points: CvPerspective,
    interpolation: int = cv.INTER_AREA,
    output_size: Tuple[int, int] = None
) -> np.ndarray:
    img_h, img_w = img.shape[0:2]

    if output_size is None:
        output_size = (img_w, img_h)

    src_points = np.float32(src_points) * np.float32((img_w, img_h)) # type: ignore

    dst_points = np.float32(dst_points) * np.float32(output_size) # type: ignore

    transform = cv.getPerspectiveTransform(src_points, dst_points)
    img = cv.warpPerspective(img, transform, output_size, flags=interpolation)

    return img


'''
def warp_perspective_keypoints_coords(
    coords: list,
    src_points: list,
    dst_points: list
) -> list:
    src_p = np.float32(src_points)
    dst_p = np.float32(dst_points)

    transform = cv.getPerspectiveTransform(src_p, dst_p)

    coords = np.float32(coords)
    coords = np.concatenate(
        (coords, np.ones(shape=(len(coords), 1), dtype=np.float32)),
        axis=1
    )

    coords = transform.dot(coords.T).T

    divide = coords[:, 2].reshape((coords.shape[0]))
    coords[:, 0:2] /= divide[:, np.newaxis]

    return coords[:, 0:2].tolist()
'''
