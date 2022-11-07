# python
import math
# project
from ganymede.math.primitives import Vector2


def square_length(vec: Vector2) -> float:
    x, y = vec

    return x ** 2 + y ** 2


def length(vec: Vector2) -> float:
    return math.sqrt(square_length(vec))


def vector_product_z(a: Vector2, b: Vector2) -> float:
    ax, ay = a
    bx, by = b

    z = ax * by - ay * bx

    return z


def scalar_product(a: Vector2, b: Vector2) -> float:
    ax, ay = a
    bx, by = b

    return ax * bx + ay * by


def cos(a: Vector2, b: Vector2) -> float:
    return scalar_product(a, b) / (length(a) * length(b) + 1e-6)


def angle(a: Vector2, b: Vector2) -> float:
    return math.acos(cos(a, b))


def sign_angle(a: Vector2, b: Vector2) -> float:
    ang = angle(a, b)

    z = vector_product_z(a, b)
    if z == 0.0:
        sign = 1
    else:
        sign = z / abs(z)

    return sign * ang


def area(a: Vector2, b: Vector2) -> float:
    return abs(vector_product_z(a, b)) / 2


def rotate(vec: Vector2, angle: float) -> Vector2:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    x, y = vec

    rot_x = cos_a * x - sin_a * y
    rot_y = sin_a * x + cos_a * y

    return rot_x, rot_y
