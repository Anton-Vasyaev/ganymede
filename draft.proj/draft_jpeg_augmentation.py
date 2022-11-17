import dependencies
# 3rd party
import numpy as np
import cv2 as cv
import ganymede.opencv as g_cv
import ganymede.core   as g_core


def aug_jpeg(img : np.ndarray, quality : float) -> np.ndarray:
    if quality < 0.0 or quality > 1.0:
        raise ValueError(f'Invalid quality value:{quality}, expected in (0.0, 1.0]')

    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), int(quality * 100)]

    result, data = cv.imencode('.jpg', img, encode_param)
    if not result:
        raise Exception(f'Failed to encode img to jpg with quality:{quality}')

    aug_img = cv.imdecode(data, cv.IMREAD_UNCHANGED)

    if aug_img is None:
        raise Exception(f'Failed to decode img from jpg with quality:{quality}')

    return aug_img


if __name__ == '__main__':
    img = g_cv.imread('./../resources/images/tank.jpg')

    g_cv.imshow('debug', img)

    jpeg_quality = 0.5

    while True:
        aug_img = aug_jpeg(img, jpeg_quality)
        key = g_cv.imshow('debug', aug_img)

        if key == 97:
            jpeg_quality = max(0.001, jpeg_quality - 0.01)
        elif key == 100:
            jpeg_quality = min(1.0, jpeg_quality + 0.01)

        print(jpeg_quality)

