# 3rd party
import cv2 as cv
import numpy as np


def blur(img : np.ndarray, kernel_size : int) -> np.ndarray:
    kernel_values_size = kernel_size * kernel_size

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= kernel_values_size

    img = cv.filter2D(img, -1, kernel)

    return img