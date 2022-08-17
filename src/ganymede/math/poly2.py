# python
from typing import List
# project
import ganymede.math.alg_tuple2 as m_t2
import ganymede.math.vec2       as m_v2
import ganymede.math.point2     as m_p2
import ganymede.math.convert    as m_convert
import ganymede.math.relation   as m_rel

from ganymede.math.point2 import Point2D


Polygon2D = List[Point2D]


def area(polygon):
    n = len(polygon)

    first_sum = 0
    y1        = polygon[0][1]
    xn        = polygon[-1][0]
    for i in range(0, n - 1):
        x_i   = polygon[i][0]
        y_ip1 = polygon[i + 1][1]

        first_sum += x_i * y_ip1

    first_sum += xn * y1

    second_sum = 0
    x1 = polygon[0][0]
    yn = polygon[-1][1]
    for i in range(0, n - 1):
        x_ip1 = polygon[i + 1][0]
        y_i   = polygon[i][1]

        second_sum += x_ip1 * y_i

    second_sum += x1 * yn

    poly_area = abs(first_sum - second_sum) / 2

    return poly_area


def contain_point(polygon, point):
    sum_of_angles = 0.0

    prev_poly_p = polygon[-1]
    for idx in range(len(polygon)):
        curr_poly_p = polygon[idx]

        prev_vec = m_t2.sub(prev_poly_p, point)
        curr_vec = m_t2.sub(curr_poly_p, point)

        sum_of_angles += m_v2.angle(prev_vec, curr_vec)

        prev_poly_p = curr_poly_p

    sum_of_angles = m_convert.rad2deg(sum_of_angles)

    return m_rel.equal_err(sum_of_angles, 360.0, 1e-2)


def rarefire_distance(polygon, distance):
    prev_p = polygon[0]

    rarefire_polygon = []
    rarefire_polygon.append(polygon[0])

    for p in polygon[1:]:
        curr_dist = m_p2.distance(prev_p, p)

        if curr_dist > distance:
            rarefire_polygon.append(p)
            prev_p = p

    return rarefire_polygon
