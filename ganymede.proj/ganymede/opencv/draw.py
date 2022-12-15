# python
from copy import deepcopy
from typing import List, cast
# 3rd party
import cv2   as cv # type: ignore
import numpy as np
# project
import ganymede.math.line2 as m_line2
import ganymede.math.bbox2 as m_bbox2
from ganymede.draw.data import *
from ganymede.math.primitives import BBox2, Point2, Line2, AlgTuple3

from ganymede.math.system_coord_transformer import SystemCoordTransformer



def draw_circle(
    img : np.ndarray,
    coord : Point2,
    color : AlgTuple3,
    radius : int = 1,
    thickness : int = 1,
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


def draw_line_p(
    img : np.ndarray,
    p1 : Point2,
    p2 : Point2,
    color : AlgTuple3,
    thickness : int = 1,
    normalize_coords : bool = True
):
    r, g, b = color
    color   = (b, g, r)

    img_h, img_w = img.shape[0:2]

    x1, y1 = p1
    x2, y2 = p2

    if normalize_coords:
        x1, y1 = x1 * img_w, y1 * img_h
        x2, y2 = x2 * img_w, y2 * img_h

    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)

    cv.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_line(
    img : np.ndarray,
    line : Line2,
    color : AlgTuple3,
    thickness : int = 1,
    normalize_coords : bool = True
):
    p1 = m_line2.first(line)
    p2 = m_line2.second(line)

    draw_line_p(img, p1, p2, color, thickness, normalize_coords)

    
    
def draw_bbox_p(
    img : np.ndarray,
    p1 : Point2,
    p2 : Point2,
    color : AlgTuple3,
    thickness : int = 1,
    normalize_coords : bool = True
):
    r, g, b = color
    color   = (b, g, r)

    img_h, img_w = img.shape[0:2]

    x1, y1 = p1
    x2, y2 = p2

    if normalize_coords:
        x1, y1 = x1 * img_w, y1 * img_h
        x2, y2 = x2 * img_w, y2 * img_h

    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)

    cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_bbox(
    img : np.ndarray,
    box : BBox2,
    color : AlgTuple3,
    thickness : int = 1,
    normalize_coords : bool = True
):
    p1 = m_bbox2.left_top(box)
    p2 = m_bbox2.right_bottom(box)

    draw_bbox_p(img, p1, p2, color, thickness, normalize_coords)


def draw_polyline(
    img : np.ndarray,
    coords : List[Point2],
    color : AlgTuple3,
    thickness : int = 1,
    normalized_coords : bool = True
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
    img : np.ndarray,
    coords : List[Point2],
    color : AlgTuple3,
    thickness : int = 1,
    normalized_coords : bool = True
):
    draw_polyline(img, coords, color, thickness, normalized_coords)

    p1 = coords[0]
    p2 = coords[-1]

    draw_line_p(img, p1, p2, color, thickness, normalized_coords)
    

def fill_polygon(
    img : np.ndarray,
    coords : List[Point2],
    color : AlgTuple3,
    normalized_coords : bool = True
):
    r, g, b = color
    color   = b, g, r

    img_h, img_w = img.shape[0:2]

    np_coords = np.array(coords)

    if normalized_coords:
        np_coords[:, 0] *= img_w
        np_coords[:, 1] *= img_h

    np_coords.shape = (1,) + np_coords.shape

    np_coords = np.int32(np_coords)

    # ToDo

    cv.fillPoly(img, np_coords, color)



def draw_canvas(
    img,
    canvas : DrawCanvas
):
    coord_t = SystemCoordTransformer(
        canvas.canvas_box,
        (0.0, 0.0, 1.0, 1.0)
    )

    for draw_shape in canvas.shapes:
        if isinstance(draw_shape, DrawLineShape):
            l_data : DrawLineShape = draw_shape
            transform_line = coord_t.transform_line(l_data.line)
            draw_line(img, transform_line, l_data.color, int(l_data.thickness))
        
        elif isinstance(draw_shape, DrawPointShape):
            p_data : DrawPointShape = draw_shape
            transform_point = coord_t.transform_point(p_data.point)
            draw_circle(
                img, 
                transform_point, 
                p_data.color, 
                max(1, int(p_data.radius - 1)), 
                int(p_data.radius)
            )
        
        elif isinstance(draw_shape, DrawPolygonShape):
            poly_data : DrawPolygonShape = draw_shape
            transform_poly = coord_t.transform_polygon(poly_data.polygon)
            draw_polygon(img, transform_poly, poly_data.color, int(poly_data.thickness))
        
        elif isinstance(draw_shape, DrawBBoxShape):
            bbox_data : DrawBBoxShape = draw_shape
            transform_bbox = coord_t.transform_bbox(bbox_data.bbox)
            draw_bbox(img, transform_bbox, bbox_data.color, int(bbox_data.thickness))
        
        elif isinstance(draw_shape, FillPolygonShape):
            fill_poly_data : FillPolygonShape = draw_shape
            transform_poly = coord_t.transform_polygon(fill_poly_data.polygon)
            fill_polygon(img, transform_poly, fill_poly_data.color)
        
        else:
            raise ValueError(f'unknown draw shape type:{type(draw_shape)}')