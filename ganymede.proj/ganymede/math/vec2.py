# python
import math
from typing import Tuple


Vector2D = Tuple[float, float]


def square_length(vec):
    x, y = vec

    return x ** 2 + y ** 2


def length(vec):
    return math.sqrt(square_length(vec))


def square_distance(a, b):
    ax, ay = a
    bx, by = b

    return (bx - ax) ** 2 + (by - ay) ** 2


def distance(a, b):
    return math.sqrt(square_distance(a, b))


def vector_product_z(a, b):
    ax, ay = a
    bx, by = b

    z = ax * by - ay * bx

    return z


def scalar_product(a, b):
    ax, ay = a
    bx, by = b

    return ax * bx + ay * by


def cos(a, b):
    return scalar_product(a, b) / (length(a) * length(b) + 1e-6)


def angle(a, b):
    return math.acos(cos(a, b))


def sign_angle(a, b):
    ang = angle(a, b)

    z    = vector_product_z(a, b)
    if z == 0.0: sign = 1
    else: sign = z / abs(z)

    return sign * ang 


def area(a, b):
    return abs(vector_product_z(a, b)) / 2


def rotate(vec, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    x, y = vec

    rot_x = cos_a * x - sin_a * y
    rot_y = sin_a * x + cos_a * y

    return (rot_x, rot_y)