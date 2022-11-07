# python
from typing import Optional
# project
from ganymede.math.primitives import Point3, Vector3, Line3


def center(line: Line3) -> Point3:
    x1, y1, z1, x2, y2, z2 = line

    c_x = (x1 + x2) / 2
    c_y = (y1 + y2) / 2
    c_z = (z1 + z2) / 2

    return c_x, c_y, c_z


def from_point_and_vec(
    point: Point3,
    vector: Vector3
) -> Line3:
    x1, y1, z1 = point
    vx, vy, vz = vector

    x2, y2, z2 = x1 + vx, y1 + vy, z1 + vz

    return x1, y1, z1, x2, y2, z2


def inter_plane(
    line: Line3,
    plane_point: Point3,
    plane_norm_vector: Vector3,
    eps: float = 1e-6
) -> Optional[Point3]:
    lx1, ly1, lz1, lx2, ly2, lz2 = line

    l_m = (lx2 - lx1)
    l_p = (ly2 - ly1)
    l_l = (lz2 - lz1)

    p_A, p_B, p_C = plane_norm_vector
    x0, y0, z0 = plane_point

    p_D = -p_A * x0 - p_B * y0 - p_C * z0

    numer = (p_D + p_A * lx1 + p_B * ly1 + p_C * lz1)
    denom = (p_A * l_m + p_B * l_p + p_C * l_l)

    if abs(numer) > eps and abs(denom) < eps:
        return None
    if abs(numer) < eps and abs(denom) < eps:
        return float('inf'), float('inf'), float('inf')

    t = - numer / denom

    x = lx1 + l_m * t
    y = ly1 + l_p * t
    z = lz1 + l_l * t

    return (x, y, z)
