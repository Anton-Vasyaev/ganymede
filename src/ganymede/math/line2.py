# python
import math
from typing import Tuple
# project
import ganymede.math.alg_tuple2 as m_t2
import ganymede.math.vec2       as m_v2
from .point2 import Point2D


Line2D = Tuple[Point2D, Point2D]


def center(line):
    (x1, y1), (x2, y2) = line

    return (x1 + x2) / 2, (y1 + y2) / 2


def left_right(line):
    (x1, _), (x2, _) = line

    min_x = min(x1, x2)
    max_x = max(x1, x2)

    return min_x, max_x


def top_bottom(line):
    (_, y1), (_, y2) = line

    min_y = min(y1, y2)
    max_y = max(y1, y2)

    return min_y, max_y


def left_right_point(line):
    p1, p2 = line
    (x1, _), (x2, _) = p1, p2

    if x1 < x2:
        return p1, p2
    else:
        return p2, p1


def top_bottom_point(line):
    p1, p2 = line
    (_, y1), (_, y2) = p1, p2

    if y1 < y2:
        return p1, p2
    else:
        return p2, p1



def bbox(line):
    l, r = left_right(line)
    t, b = top_bottom(line)

    return l, t, r, b


def equation(line):
    (x1, y1), (x2, y2) = line

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    return a, b, c


def square_length(line):
    (x1, y1), (x2, y2) = line

    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def length(line):
    return math.sqrt(square_length(line))


def crossing(a, b, eps=1e-8):
    (x1, y1), (x2, y2) = a
    (x3, y3), (x4, y4) = b

    x_num = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    y_num = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    
    div = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    x = x_num / (div + eps)
    y = y_num / (div + eps)

    return x, y


def distance_to_point(line, point, eps=1e-8):
    x, y = point

    a, b, c = equation(line)

    denominator = abs(a * x + b * y + c)
    divider     = math.sqrt(a ** 2 + b ** 2)

    return denominator / (divider + eps)


def near_point_on_line(line, point, eps=1e-8):
    x0, y0 = point

    a, b, c = equation(line)

    divider = a * a + b * b

    x = (b * ( b * x0 - a * y0) - a * c) / (divider + eps)
    y = (a * (-b * x0 + a * y0) - b * c) / (divider + eps)

    return x, y


def lie_on_line(line, point, relative_eps=1e-4, eps=1e-8):
    near_point = near_point_on_line(line, point, eps)

    eps = length(line) * relative_eps

    dist_vector = m_t2.sub(near_point, point)

    return m_v2.length(dist_vector) < eps


def lie_on_segment(line, point, relative_eps=1e-4, eps=1e-8):
    p1 = line[0]
    p2 = line[1]

    if not lie_on_line(line, point, relative_eps, eps):
        return False

    sqr_len = square_length(line)

    sqr_dist_1 = m_v2.square_length(m_t2.sub(point, p1))
    sqr_dist_2 = m_v2.square_length(m_t2.sub(point, p2))

    return sqr_dist_1 <= sqr_len and sqr_dist_2 <= sqr_len


def near_point_on_segment(line, point, relative_eps=1e-4, eps=1e-8):
    p1 = line[0]
    p2 = line[1]

    n_p = near_point_on_line(line, point, eps)

    if lie_on_segment(line, n_p, relative_eps, eps):
        return n_p

    sqr_dist_1 = m_v2.square_length(m_t2.sub(n_p, p1))
    sqr_dist_2 = m_v2.square_length(m_t2.sub(n_p, p2))

    return p1 if sqr_dist_2 > sqr_dist_1 else p2


def segment_distance_to_point(line, point, relative_eps=1e-4, eps=1e-8):
    n_p = near_point_on_segment(line, point, relative_eps, eps)

    return m_v2.length(m_t2.sub(point, n_p))