# python
from typing import List
# project
import ganymede.math.alg_tuple2 as m_t2
import ganymede.math.vec2 as m_v2
import ganymede.math.point2 as m_p2
import ganymede.math.convert as m_convert
import ganymede.math.relation as m_rel
from ganymede.math.primitives import Polygon2, Point2, BBox2


def area(polygon: Polygon2) -> float:
    n = len(polygon)

    first_sum = 0.0
    y1 = polygon[0][1]
    xn = polygon[-1][0]
    for i in range(0, n - 1):
        x_i = polygon[i][0]
        y_ip1 = polygon[i + 1][1]

        first_sum += x_i * y_ip1

    first_sum += xn * y1

    second_sum = 0.0
    x1 = polygon[0][0]
    yn = polygon[-1][1]
    for i in range(0, n - 1):
        x_ip1 = polygon[i + 1][0]
        y_i = polygon[i][1]

        second_sum += x_ip1 * y_i

    second_sum += x1 * yn

    poly_area = abs(first_sum - second_sum) / 2

    return poly_area


def contain_point(polygon: Polygon2, point: Point2) -> bool:
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


def rarefire_distance(polygon: Polygon2, distance: float) -> Polygon2:
    prev_p = polygon[0]

    rarefire_polygon = []
    rarefire_polygon.append(polygon[0])

    for p in polygon[1:]:
        curr_dist = m_p2.distance(prev_p, p)

        if curr_dist > distance:
            rarefire_polygon.append(p)
            prev_p = p

    return rarefire_polygon


def normalize_on_contour(polygon : Polygon2, contour : BBox2) -> Polygon2:
    norm_poly = []
    for p in polygon:
        norm_poly.append(m_p2.normalize_on_contour(p, contour))
        
    return norm_poly



def get_interpolate_position_on_polyline(
    polyline : List[Point2],
    interpolate_value : float
) -> Point2:
    if len(polyline) < 2:
        raise ValueError(f'len of polyline < 2:{len(polyline)}')
    
    if interpolate_value < 0.0 or interpolate_value > 1.0:
        raise ValueError(f'invalid range of norm_len:{interpolate_value} (requried [0.0, 1.0]).')
    
    
    full_len = 0.0
    
    prev_p = polyline[0]
    for p in polyline[1:]:
        full_len += m_p2.distance(p, prev_p)
        prev_p = p
        
    move_len = 0.0
    requred_dist = interpolate_value * full_len
    prev_p = polyline[0]
    for p in polyline:
        current_len = m_p2.distance(p, prev_p)
        if move_len + current_len > requred_dist:
            vec_len = requred_dist - move_len
            
            if vec_len == 0.0:
                return prev_p
            
            vec_point = m_t2.sub(p, prev_p)
            vec_point = m_v2.normalize(vec_point, vec_len)
            
            calc_p = m_t2.add(prev_p, vec_point)
            
            return calc_p
            
        prev_p = p
        move_len += current_len
            

    return polyline[-1]