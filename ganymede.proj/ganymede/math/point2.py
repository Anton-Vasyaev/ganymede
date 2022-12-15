# python
import math
from typing import List
# project
from ganymede.math.functions import clip
from ganymede.math.primitives import Point2, BBox2, Mat3x3


def square_distance(a: Point2, b: Point2):
    ax, ay = a
    bx, by = b

    return (ax - bx) ** 2 + (ay - by) ** 2


def distance(a: Point2, b: Point2):
    return math.sqrt(square_distance(a, b))


def get_contour(points: List[Point2]) -> BBox2:
    first_point = points[0]

    left = first_point[0]
    right = first_point[0]
    top = first_point[1]
    bottom = first_point[1]

    for point in points[1:]:
        x, y = point
        left = min(x, left)
        right = max(x, right)
        top = min(y, top)
        bottom = max(y, bottom)

    return (left, top, right, bottom)


def normalize_on_contour(point: Point2, bbox: BBox2) -> Point2:
    x, y = point
    l, t, r, b = bbox
    w, h = r - l, b - t

    x, y = (x - l) / w, (y - t) / h

    return x, y


def reverse_normalize_on_contour(point: Point2, bbox: BBox2) -> Point2:
    x, y = point
    l, t, r, b = bbox
    w, h = r - l, b - t

    x, y = x * w, y * h

    x, y = x + l, y + t

    return x, y


def normalize_on_self(points: List[Point2]) -> List[Point2]:
    l, t, r, b = get_contour(points)

    w, h = r - l, b - t

    normalized_points = []
    for p in points:
        x, y = p

        x, y = (x - l) / w, (y - t) / h

        normalized_points.append((x, y))

    return normalized_points


def rotate(
    point: Point2,
    angle: float,
    anchor: Point2 = (0.0, 0.0)
) -> Point2:
    a_x, a_y = anchor

    sin = math.sin(angle)
    cos = math.cos(angle)

    x, y = point
    x, y = x - a_x, y - a_y

    rot_x = x * cos - y * sin
    rot_y = x * sin + y * cos

    rot_x += a_x
    rot_y += a_y

    return rot_x, rot_y


def perspective_transform(
    point: Point2,
    mat: Mat3x3
) -> Point2:
    x, y = point

    xt = mat[0][0] * x + mat[0][1] * y + mat[0][2]
    yt = mat[1][0] * x + mat[1][1] * y + mat[1][2]
    t = mat[2][0] * x + mat[2][1] * y + mat[2][2]

    x = xt / t
    y = yt / t

    return x, y


def contour_clip(point: Point2, bbox: BBox2) -> Point2:
    x, y = point

    l, t, r, b = bbox

    return clip(x, l, r), clip(y, t, b)
