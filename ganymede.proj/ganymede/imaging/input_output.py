# 3rd party
import cv2 as cv
import numpy as np


def encode_img_to_png(img : np.ndarray) -> bytearray:
    ret, png_data = cv.imencode('.png', img)

    if png_data is None: raise Exception('failed to encode img to png format')

    return bytearray(png_data)


def decode_img(binary_data : bytes) -> np.ndarray:
    data = np.frombuffer(binary_data, dtype=np.uint8)

    img : np.ndarray = cv.imdecode(data, cv.IMREAD_UNCHANGED)
    if img is None: raise Exception('failed to decode img')

    return img