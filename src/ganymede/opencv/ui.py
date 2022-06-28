# 3rd party
import cv2   as cv
import numpy as np


def imshow(
    window_name, 
    img, 
    wait_ms = 0,
    normalized_float = True,
    escape_catch=True
):
    if np.issubdtype(img.dtype, np.floating):
            if not normalized_float:
                img = img / 255.0
    cv.imshow(window_name, img)
    key = cv.waitKey(wait_ms)
    
    if escape_catch:
        if key == 27:
            exit()

    return key