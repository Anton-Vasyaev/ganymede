# python
import math
from typing import Tuple
# project
import ganymede.math.vec2     as vec2
import ganymede.math.relation as m_rel


Vector3D = Tuple[float, float, float]


def square_length(vec):
    x, y, z = vec

    return x ** 2 + y ** 2 + z ** 2


def length(vec):
    return math.sqrt(square_length(vec))


def normalize(vec, normalize_length = 1.0):
    scale = normalize_length / (length(vec) + 1e-6)

    x, y, z = vec
    x, y, z = x * scale, y * scale, z * scale

    return x, y, z


def vector_product(a, b):
    ax, ay, az = a
    bx, by, bz = b

    x = ay * bz - az * by
    y = az * bx - ax * bz
    z = ax * by - ay * bx

    return x, y, z


def scalar_product(a, b):
    ax, ay, az = a
    bx, by, bz = b

    return ax * bx + ay * by + az * bz


def area(a, b):
    return length(vector_product(a, b)) / 2


def sin(a, b):
    return area(a, b) / (length(a) * length(b) + 1e-6)


def cos(a, b):
    return scalar_product(a, b) / (length(a) * length(b) + 1e-6)


def angle(a, b):
    return math.acos(cos(a, b))


def angle_axis(a, b, main_axis, rel_eps = 1e-8):
    if main_axis == 0:
        x_idx, y_idx = 1, 2
    elif main_axis == 1:
        x_idx, y_idx = 2, 0
    elif main_axis == 2:
        x_idx, y_idx = 0, 1

    a2d = a[x_idx], a[y_idx]
    b2d = b[x_idx], b[y_idx]

    if m_rel.less_to_grt(vec2.length(a2d), vec2.length(b2d)) < rel_eps: return 0.0
    
    return vec2.sign_angle(a2d, b2d)


def angle_x_axis(a, b, rel_ep = 1e-8):
    return angle_axis(a, b, 0, rel_ep)


def angle_y_axis(a, b, rel_ep = 1e-8):
    return angle_axis(a, b, 1, rel_ep)


def angle_z_axis(a, b, rel_eps = 1e-8):
    return angle_axis(a, b, 2, rel_eps)


def axis_angles(a, b, rel_eps = 1e-8):
    z_angle = angle_z_axis(a, b, rel_eps)
    b = rotate_z_axis(b, -z_angle)

    y_angle = angle_y_axis(a, b, rel_eps)
    b = rotate_y_axis(b, y_angle)

    x_angle = angle_x_axis(a, b, rel_eps)

    return x_angle, y_angle, z_angle


def rotate_x_axis(vec, angle):
    x, y, z = vec
    y, z    = vec2.rotate((y, z), angle)

    return x, y, z


def rotate_y_axis(vec, angle):
    x, y, z = vec
    z, x    = vec2.rotate((z, x), angle)

    return x, y, z


def rotate_z_axis(vec, angle):
    x, y, z = vec
    x, y    = vec2.rotate((x, y), angle)

    return x, y, z


def rotate(vec, x_angle, y_angle, z_angle):
    vec = rotate_x_axis(vec, x_angle)
    vec = rotate_y_axis(vec, y_angle)
    vec = rotate_z_axis(vec, z_angle)

    return vec


def reverse_rotate(vec, x_angle, y_angle, z_angle):
    x_angle = -x_angle
    y_angle = -y_angle
    z_angle = -z_angle
    
    vec = rotate_z_axis(vec, z_angle)
    vec = rotate_y_axis(vec, y_angle)
    vec = rotate_x_axis(vec, x_angle)

    return vec