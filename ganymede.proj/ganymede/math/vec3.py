# python
import math
# project
import ganymede.math.vec2 as vec2
import ganymede.math.relation as m_rel
from ganymede.math.primitives import Vector3, AlgTuple3


def square_length(vec: Vector3) -> float:
    x, y, z = vec

    return x ** 2 + y ** 2 + z ** 2


def length(vec: Vector3) -> float:
    return math.sqrt(square_length(vec))


def normalize(vec: Vector3, normalize_length: float = 1.0) -> Vector3:
    scale = normalize_length / (length(vec) + 1e-6)

    x, y, z = vec
    x, y, z = x * scale, y * scale, z * scale

    return x, y, z


def vector_product(a: Vector3, b: Vector3) -> Vector3:
    ax, ay, az = a
    bx, by, bz = b

    x = ay * bz - az * by
    y = az * bx - ax * bz
    z = ax * by - ay * bx

    return x, y, z


def scalar_product(a: Vector3, b: Vector3) -> float:
    ax, ay, az = a
    bx, by, bz = b

    return ax * bx + ay * by + az * bz


def area(a: Vector3, b: Vector3) -> float:
    return length(vector_product(a, b)) / 2


def sin(a: Vector3, b: Vector3) -> float:
    return area(a, b) / (length(a) * length(b) + 1e-6)


def cos(a: Vector3, b: Vector3) -> float:
    return scalar_product(a, b) / (length(a) * length(b) + 1e-6)


def angle(a: Vector3, b: Vector3) -> float:
    return math.acos(cos(a, b))


def angle_on_custom_axis(a: Vector3, b: Vector3, axis_idx: float, rel_eps: float = 1e-8) -> float:
    if axis_idx == 0:
        x_idx, y_idx = 1, 2
    elif axis_idx == 1:
        x_idx, y_idx = 2, 0
    elif axis_idx == 2:
        x_idx, y_idx = 0, 1

    a2d = a[x_idx], a[y_idx]
    b2d = b[x_idx], b[y_idx]

    if m_rel.less_to_grt(vec2.length(a2d), vec2.length(b2d)) < rel_eps:
        return 0.0

    return vec2.sign_angle(a2d, b2d)


def angle_on_x_axis(a: Vector3, b: Vector3, rel_eps: float = 1e-8) -> float:
    return angle_on_custom_axis(a, b, 0, rel_eps)


def angle_on_y_axis(a: Vector3, b: Vector3, rel_eps: float = 1e-8) -> float:
    return angle_on_custom_axis(a, b, 1, rel_eps)


def angle_on_z_axis(a: Vector3, b: Vector3, rel_eps: float = 1e-8) -> float:
    return angle_on_custom_axis(a, b, 2, rel_eps)


def rotate_on_x_axis(vec: Vector3, angle: float) -> Vector3:
    x, y, z = vec
    y, z = vec2.rotate((y, z), angle)

    return x, y, z


def rotate_on_y_axis(vec: Vector3, angle: float) -> Vector3:
    x, y, z = vec
    z, x = vec2.rotate((z, x), angle)

    return x, y, z


def rotate_on_z_axis(vec: Vector3, angle: float) -> Vector3:
    x, y, z = vec
    x, y = vec2.rotate((x, y), angle)

    return x, y, z


def get_axes_angles_to_rotate(a: Vector3, b: Vector3, rel_eps: float = 1e-8) -> AlgTuple3:
    z_angle = angle_on_z_axis(a, b, rel_eps)
    b = rotate_on_z_axis(b, -z_angle)

    y_angle = angle_on_y_axis(a, b, rel_eps)
    b = rotate_on_y_axis(b, y_angle)

    x_angle = angle_on_x_axis(a, b, rel_eps)

    return x_angle, y_angle, z_angle


def rotate(vec: Vector3, x_angle: float, y_angle: float, z_angle: float) -> Vector3:
    vec = rotate_on_x_axis(vec, x_angle)
    vec = rotate_on_y_axis(vec, y_angle)
    vec = rotate_on_z_axis(vec, z_angle)

    return vec


def reverse_rotate(vec: Vector3, x_angle: float, y_angle: float, z_angle: float) -> Vector3:
    x_angle = -x_angle
    y_angle = -y_angle
    z_angle = -z_angle

    vec = rotate_on_z_axis(vec, z_angle)
    vec = rotate_on_y_axis(vec, y_angle)
    vec = rotate_on_x_axis(vec, x_angle)

    return vec
