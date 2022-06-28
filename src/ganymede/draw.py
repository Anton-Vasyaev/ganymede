# python
from copy import deepcopy
# 3rd party
import numpy as np
import cv2   as cv
from   PIL   import ImageFont, ImageDraw, Image


GRAY_RED   = 0.299
GRAY_GREEN = 0.587
GRAY_BLUE  = 0.114


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


def draw_point_list(
    img,
    points,
    color,
    radius = 2,
    thickness = 2,
    normalize_coords = True
):
    for point in points: draw_point(img, point, color, radius, thickness, normalize_coords)


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


def draw_rectangle_bbox(
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


def draw_polyline_list(
    img,
    coords_list,
    color,
    thickness=1,
    normalize_coords = True
):
    for coords in coords_list:
        draw_polyline(
            img, coords, color, thickness, normalize_coords
        )


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


def draw_polygon_list(
    img,
    coords_list,
    color,
    thickness=1,
    normalize_coords = True
):
    for coords in coords_list:
        draw_polygon(img, coords, color, thickness, normalize_coords)


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


def fill_polygon_list(
    img,
    coords_list,
    color,
    normalized_coords = True
):
    for coords in coords_list:
        fill_polygon(img, coords, color, normalized_coords)


def draw_text_list(
    img,
    text_list,
    font_size=0.05
):
    is_gray_img_flag = False

    if len(img.shape) == 3:
        if img.shape[2] == 1:
            img.shape = img.shape[:2]
            is_gray_img_flag = True
    elif len(img.shape) == 2:
        is_gray_img_flag = True

    img_h, img_w = img.shape[0:2]

    font_size = int(font_size * img_h)

    font = ImageFont.truetype("arial.ttf", font_size)

    if np.issubdtype(img.dtype, np.floating):
        copy_img = (img * 255).astype(np.uint8)
    else:
        copy_img = img

    img_pil = Image.fromarray(copy_img)
    draw    = ImageDraw.Draw(img_pil)

    for text_block in text_list:
        text, left_corner, color = text_block

        x, y    = left_corner
        x, y    = int(x * img_w), int(y * img_h)
        r, g, b = color

        real_color = (b, g, r)
        if is_gray_img_flag:
            real_color = int(r * GRAY_RED + g * GRAY_GREEN + b * GRAY_BLUE)

        draw.text((x, y), text, font=font, fill=real_color)

    draw_img = np.array(img_pil)

    img[:] = draw_img