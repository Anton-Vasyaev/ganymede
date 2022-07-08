# python
import math
from typing import Tuple


Point3D = Tuple[float, float, float]


def xy(vec):
    x, y, z = vec
    return x, y


def yx(vec):
    x, y, z = vec
    return y, x


def xz(vec):
    x, y, z = vec
    return x, z


def zx(vec):
    x, y, z = vec
    return z, x


def yz(vec):
    x, y, z = vec
    return y, z


def zy(vec):
    x, y, z = vec
    return z, y


def xzy(vec):
    x, y, z = vec
    return x, z, y


def yxz(vec):
    x, y, z = vec
    return y, x, z


def yzx(vec):
    x, y, z = vec
    return y, z, x


def zxy(vec):
    x, y, z = vec
    return z, x, y


def zyx(vec):
    x, y, z = vec
    return z, y, x


def square_distance(a, b):
    ax, ay, az = a
    bx, by, bz = b

    return (bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2


def distance(a, b):
    return math.sqrt(square_distance(a, b))