# python
from typing import Union, Tuple, cast
from numbers import Number
# project
import ganymede.math.auxiliary as m_aux
import ganymede.math.point2 as m_p2
from ganymede.math.primitives import BBox2, Point2, Size2


def width(bbox: BBox2) -> float:
    x1, y1, x2, y2 = bbox

    return x2 - x1


def height(bbox: BBox2) -> float:
    x1, y1, x2, y2 = bbox

    return y2 - y1


def left_top(bbox: BBox2) -> Point2:
    return bbox[0], bbox[1]


def left_bottom(bbox: BBox2) -> Point2:
    return bbox[1], bbox[3]


def right_bottom(bbox: BBox2) -> Point2:
    return bbox[2], bbox[3]


def right_top(bbox: BBox2) -> Point2:
    return bbox[2], bbox[1]


def from_points(left_top: Point2, right_bottom: Point2) -> BBox2:
    return left_top[0], left_top[1], right_bottom[0], right_bottom[1]


def clip(
    bbox: BBox2,
    min_val: Union[float, Size2],
    max_val: Union[float, Size2]
):
    if isinstance(min_val, Number):
        min_val_num = cast(float, min_val)
        max_val_num = cast(float, max_val)
        min_w, min_h = min_val_num, min_val_num
        max_w, max_h = max_val_num, max_val_num
    else:
        min_val_size = cast(Size2, min_val)
        max_val_size = cast(Size2, max_val)
        min_w, min_h = min_val_size
        max_w, max_h = max_val_size

    x1, y1, x2, y2 = bbox

    x1 = cast(float, m_aux.clip(x1, min_w, max_w))
    y1 = cast(float, m_aux.clip(y1, min_h, max_h))

    x2 = cast(float, m_aux.clip(x2, min_w, max_w))
    y2 = cast(float, m_aux.clip(y2, min_h, max_h))

    return x1, y1, x2, y2


def scale(
    bbox: BBox2,
    scale_size: Union[float, Size2]
):
    if isinstance(scale_size, Number):
        scale_w, scale_h = cast(float, scale_size), cast(float, scale_size)
    else:
        scale_size_s = cast(Size2, scale_size)
        scale_w, scale_h = scale_size_s

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    xc, yc = x1 + w / 2, y1 + h / 2

    w *= scale_w
    h *= scale_h

    x1, y1 = xc - w / 2, yc - w / 2
    x2, y2 = xc + w / 2, yc + w / 2

    return [x1, y1, x2, y2]


def area(bbox: BBox2) -> float:
    l, t, r, b = bbox

    return abs(r - l) * abs(b - t)


def center(bbox: BBox2) -> Point2:
    l, t, r, b = bbox

    return (l + r) / 2, (t + b) / 2


def iom(v: BBox2, z: BBox2) -> float:
    vx1, vy1, vx2, vy2 = v
    zx1, zy1, zx2, zy2 = z

    l = max(vx1, zx1)
    r = min(vx2, zx2)
    t = max(vy1, zy1)
    b = min(vy2, zy2)

    if l >= r or t >= b:
        return 0.0

    a_area = area(v)
    b_area = area(z)
    min_area = min(a_area, b_area)

    inter_area = area((l, t, r, b))

    return inter_area / min_area


def corner_points(bbox: BBox2) -> Tuple[Point2, Point2, Point2, Point2]:
    x1, y1, x2, y2 = bbox

    return (
        (x1, y1),
        (x2, y1),
        (x1, y2),
        (x2, y2)
    )


def normalize_on_contour(src_bbox: BBox2, contour: BBox2) -> BBox2:
    p1 = left_top(src_bbox)
    p2 = right_bottom(src_bbox)

    p1 = m_p2.normalize_on_contour(p1, contour)
    p2 = m_p2.normalize_on_contour(p2, contour)

    return from_points(p1, p2)


def reverse_normalize_on_contour(src_bbox: BBox2, contour: BBox2) -> BBox2:
    p1 = left_top(src_bbox)
    p2 = right_bottom(src_bbox)

    p1 = m_p2.reverse_normalize_on_contour(p1, contour)
    p2 = m_p2.reverse_normalize_on_contour(p2, contour)

    return from_points(p1, p2)
