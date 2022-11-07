# python
from typing import List
# 3rd party
import cv2   as cv # type: ignore
import numpy as np
# project
from ganymede.math.primitives import Line2, Polygon2


def hough_lines_p(
    img             : np.ndarray, 
    rho             : float, 
    theta           : float, 
    threshold       : int, 
    min_line_length : int = 0,
    max_line_gap    : int = 0
) -> List[Line2]:
    img_h, img_w = img.shape[0:2]

    lines = cv.HoughLinesP(
        img, 
        rho, 
        theta, 
        threshold, 
        min_line_length, 
        max_line_gap
    )

    new_lines : List[Line2] = []
    for line in lines:
        line = line[0]

        p1 = (line[0], line[1])
        p2 = (line[2], line[3])

        x1, y1 = p1
        x2, y2 = p2

        x1, y1 = x1 / img_w, y1 / img_h
        x2, y2 = x2 / img_w, y2 / img_h

        new_lines.append((x1, y1, x2, y2))

    return new_lines


def find_contours(
    img    : np.ndarray, 
    mode   : int, 
    method : int
) -> List[Polygon2]:
    img_h, img_w = img.shape[0:2]
    contours, hierarchy = cv.findContours(img, mode, method)

    new_contours = []

    for contour in contours:
        contour = contour.tolist()
        new_contour = []
        for p in contour:
            x, y = p[0]
            x, y = x / img_w, y / img_h

            new_contour.append((x, y))

        new_contours.append(new_contour)

    return new_contours
    