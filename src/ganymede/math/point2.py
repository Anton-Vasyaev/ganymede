# python
import math
# project
import ganymede.math.poly2 as poly2
import ganymede.math.vec2  as vec2


def square_distance(a, b):
    ax, ay = a
    bx, by = b

    return (ax - bx) ** 2 + (ay - by) ** 2


def distance(a, b):
    return math.sqrt(square_distance(a, b))


def get_bbox(points):
    first_points = points[0]

    left  = first_points[0]
    right = first_points[0]
    top    = first_points[1]
    bottom = first_points[1]

    for point in points[1:]:
        x, y   = point
        left   = min(x, left)
        right  = max(x, right)
        top    = min(y, top)
        bottom = max(y, bottom)

    return [left, top, right, bottom]


def normalize_bbox(point, bbox):
    x, y = point
    l, t, r, b = bbox
    w, h = r - l, b - t

    x, y = (x - l) / w, (y - t) / h

    return x, y


def reverse_normalize_bbox(point, bbox):
    x, y       = point
    l, t, r, b = bbox
    w, h       = r - l, b - t

    x, y = x * w, y * h

    x, y = x + l, y + t

    return x, y


def normalize_on_self(points):
    l, t, r, b = get_bbox(points)

    w, h = r - l, b - t

    normalized_points = []
    for p in points:
        x, y = p

        x, y = (x - l) / w, (y - t) / h

        normalized_points.append((x, y))

    return normalized_points


def rotate(point, angle, anchor = (0.0, 0.0)):
    a_x, a_y = anchor

    sin = math.sin(angle)
    cos = math.cos(angle)

    x, y = point
    x, y = x - a_x, y - a_y

    x = x * cos - y * sin
    y = x * sin + y * cos

    return x, y


def rotate_list(points, angle, anchor = (0.0, 0.0)):
    # ToDo need to vectorize via numpy in the future
    return [ rotate(p, angle, anchor) for p in points]


def perspective_transform(
    point,
    mat
):
    x, y = point
    
    xt = mat[0][0] * x + mat[0][1] * y + mat[0][2]
    yt = mat[1][0] * x + mat[1][1] * y + mat[1][2]
    t  = mat[2][0] * x + mat[2][1] * y + mat[2][2]
    
    x = xt / t
    y = yt / t
    
    return x, y
