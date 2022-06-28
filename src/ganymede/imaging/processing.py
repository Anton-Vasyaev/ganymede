# python
from typing import Tuple
# 3rd party
import cv2 as cv
import numpy as np


def create_channel_if_not_exist(img : np.ndarray) -> None:
    if len(img.shape) == 2:
        img.shape = img.shape + (1,)


def cast_one_channel_img(
    img :      np.ndarray, 
    channels : int = 3
) -> np.ndarray:
    if len(img.shape) == 3:
        if img.shape[2] != 1:
            raise ValueError('img channels != 1')

    img = img.copy()
    create_channel_if_not_exist(img)
    img = np.repeat(img, channels, axis=2)

    return img


def drop_alpha_channel(img : np.ndarray) -> np.ndarray:
    if len(img.shape) != 3:
        raise ValueError('img has no channels')

    if img.shape[2] != 4:
        raise ValueError(f'bgra image has only 4 channels')
        
    return img[...,0:3]


def roi_ltrb(
    img        : np.ndarray, 
    roi_coords : list, 
    normalized : bool = True
) -> np.ndarray:
    img_h, img_w = img.shape[0:2]

    x_max = 1.0
    y_max = 1.0

    left, top, right, bottom = roi_coords

    if not normalized: x_max, y_max = x_max * img_w, y_max * img_h

    valid_border = left >= right or top  >= bottom

    valid_left   = 0 >  left     or left   >= x_max
    valid_right  = 0 >= right    or right  >  x_max
    valid_top    = 0 >  top      or top    >= y_max
    valid_bottom = 0 >= bottom   or bottom >  y_max

    valid = valid_border or valid_left or valid_right or valid_top or valid_bottom
    if valid: raise ValueError(f'invalid roi:{[left, top, right, bottom]} on area:[{(x_max, y_max)}]')

    if normalized:
        left, top     = left  * img_w, top    * img_h
        right, bottom = right * img_w, bottom * img_h

    left,  top    = int(left),  int(top)
    right, bottom = int(right), int(bottom)

    return img[top:bottom, left:right]


def roi_rect(
    img        : np.ndarray,
    roi_coords : list,
    normalized : bool = True
) -> np.ndarray:
    x1, y1, w, h = roi_coords

    x2, y2 = x1 + w, y1 + h

    return roi_ltrb(img, [x1, y1, x2, y2], normalized)

