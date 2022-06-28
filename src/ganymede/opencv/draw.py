# python
from copy import deepcopy
# 3rd party
import cv2   as cv
import numpy as np



def draw_point(
    img,
    coord,
    color,
    radius    = 2,
    thickness = 2,
    normalize_coords = True
):
    r, g, b = color
    color   = (b, g, r)
    
    img_h, img_w = img.shape[0:2]

    x, y = coord
    if normalize_coords: x, y = x * img_w, y * img_h
    x, y = int(x), int(y)

    r, g, b = color

    cv.circle(img, (x, y), radius, color, thickness)
    
    
def draw_line(
    img,
    p1,
    p2,
    color,
    thickness
):
    r, g, b = color
    color   = (b, g, r)

    img_h, img_w = img.shape[0:2]

    x1, y1 = p1
    x2, y2 = p2

    x1, y1 = int(x1 * img_w), int(y1 * img_h)
    x2, y2 = int(x2 * img_w), int(y2 * img_h)

    cv.line(img, (x1, y1), (x2, y2), color, thickness)
    
    
def draw_rectangle_p(
    img,
    p1,
    p2,
    color,
    thickness = 2
):
    r, g, b = color
    color   = (b, g, r)

    img_h, img_w = img.shape[0:2]

    x1, y1 = p1
    x2, y2 = p2

    x1, y1 = int(x1 * img_w), int(y1 * img_h)
    x2, y2 = int(x2 * img_w), int(y2 * img_h)

    cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(
    img,
    rectangle,
    color,
    thickness = 2
):
    x1, y1, w, h = rectangle
    x2, y2 = x1 + w, y1 + h

    draw_rectangle_p(img, (x1, y1), (x2, y2), color, thickness)


def draw_bbox(
    img,
    bbox,
    color,
    thickness = 2
):
    x1, y1, x2, y2 = bbox

    draw_rectangle_p(img, (x1, y1), (x2, y2), color, thickness)


def draw_polyline(
    img,
    coords,
    color,
    thickness = 1,
    normalized_coords = True
):
    r, g, b = color
    color   = b, g, r

    img_h, img_w = img.shape[0:2]

    prev_coord = coords[0]
    for coord in coords[1:]:
    
        x1, y1 = prev_coord
        x2, y2 = coord

        if normalized_coords:
            x1, y1 = x1 * img_w, y1 * img_h
            x2, y2 = x2 * img_w, y2 * img_h

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        prev_coord = coord

        cv.line(img, (x1, y1), (x2, y2), color, thickness)
        
        
def draw_polygon(
    img,
    coords,
    color,
    thickness=1,
    normalize_coords = True
):
    coords = deepcopy(coords)
    coords = list(coords) if type(coords) != list else coords
      
    coords.append(coords[0])

    draw_polyline(img, coords, color, thickness, normalize_coords)
    

def fill_polygon(
    img,
    coords,
    color,
    normalized_coords = True
):
    r, g, b = color
    color   = b, g, r

    img_h, img_w = img.shape[0:2]

    coords = np.array(coords)

    if normalized_coords:
        coords[:, 0] *= img_w
        coords[:, 1] *= img_h
    coords = np.array([coords], dtype=np.int32)

    cv.fillPoly(img, coords, color)