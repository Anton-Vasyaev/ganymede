# python 
from typing import List, Tuple
# 3rd party 
import cv2   as cv # type: ignore
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


def imshow_multi(
    targets          : List[Tuple[str, np.ndarray]],
    wait_ms          : int  = 0,
    normalized_float : bool = True,
    escape_catch     : bool = True
):
    for window_name, img in targets:
        if np.issubdtype(img.dtype, np.floating):
            if not normalized_float:
                img = img / 255.0

        cv.imshow(window_name, img)

    key = cv.waitKey(wait_ms)
    if escape_catch:
        if key == 27:
            exit()

    return key