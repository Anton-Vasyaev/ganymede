# python
import math
# project
from ganymede.math.primitives import Point3, Point2


def xy(p: Point3) -> Point2:
    x, y, z = p
    return x, y


def yx(p: Point3) -> Point2:
    x, y, z = p
    return y, x


def xz(p: Point3) -> Point2:
    x, y, z = p
    return x, z


def zx(p: Point3) -> Point2:
    x, y, z = p
    return z, x


def yz(p: Point3) -> Point2:
    x, y, z = p
    return y, z


def zy(p: Point3) -> Point2:
    x, y, z = p
    return z, y


def xzy(p: Point3) -> Point3:
    x, y, z = p
    return x, z, y


def yxz(p: Point3) -> Point3:
    x, y, z = p
    return y, x, z


def yzx(p: Point3) -> Point3:
    x, y, z = p
    return y, z, x


def zxy(p: Point3) -> Point3:
    x, y, z = p
    return z, x, y


def zyx(p: Point3) -> Point3:
    x, y, z = p
    return z, y, x


def square_distance(a: Point3, b: Point3) -> float:
    ax, ay, az = a
    bx, by, bz = b

    return (bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2


def distance(a: Point3, b: Point3) -> float:
    return math.sqrt(square_distance(a, b))
