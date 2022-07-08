# python
from typing import Tuple
# project
from .point3 import Point3D


Line3D = Tuple[Point3D, Point3D]


def center(line):
    p1, p2 = line
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    c_x = (x1 + x2) / 2
    c_y = (y1 + y2) / 2
    c_z = (z1 + z2) / 2

    return c_x, c_y, c_z


def from_point_and_vec(point, vector):
    x1, y1, z1 = point
    vx, vy, vz = vector

    x2, y2, z2 = x1 + vx, y1 + vy, z1 + vz

    return (x1, y1, z1), (x2, y2, z2)



def inter_plane(line, plane_point, plane_norm_vector, eps = 1e-6):
    lp1, lp2 = line
    lx1, ly1, lz1 = lp1
    lx2, ly2, lz2 = lp2

    l_m = (lx2 - lx1)
    l_p = (ly2 - ly1)
    l_l = (lz2 - lz1)

    p_A, p_B, p_C = plane_norm_vector
    x0, y0, z0 = plane_point

    p_D = -p_A * x0 - p_B * y0 - p_C * z0

    numer = (p_D + p_A * lx1 + p_B * ly1 + p_C * lz1)
    denom = (p_A * l_m + p_B * l_p + p_C * l_l)

    if abs(numer) > eps and abs(denom) < eps: return None
    if abs(numer) < eps and abs(denom) < eps: return float('inf')

    t = - numer / denom

    x = lx1 + l_m * t
    y = ly1 + l_p * t
    z = lz1 + l_l * t

    return (x, y, z)