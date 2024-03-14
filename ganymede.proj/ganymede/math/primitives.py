# python
from typing import Tuple, List, Any
from numbers import Number


AlgTuple2 = Tuple[float, float]
''' Algebraic tuple, contain two elements. '''

AlgTuple3 = Tuple[float, float, float]
''' Algebraic tuple, contain three elements. '''

AlgTuple4 = Tuple[float, float, float]
''' Algebraic tuple, contain four elements. '''

Point2 = AlgTuple2
''' Alias for `AlgTuple2`. Represent two-dimensional coordinate. '''

Vector2 = AlgTuple2
''' Alias for `AlgTuple2`. Represent two-dimensional vector. '''

Size2 = AlgTuple2
''' Alias for `AlgTuple2`. Represent two-dimensional size. '''

Point3 = AlgTuple3
''' Alias for `AlgTuple3`. Represent three-dimensional coordinate. '''

Vector3 = AlgTuple3
''' Alias for `AlgTuple3`. Represent three-dimensional vector. '''

Size3 = AlgTuple3
''' Alias for `AlgTuple3`. Represent three-dimensional size. '''

Color3 = AlgTuple3
''' Alias for `AlgTuple3`. Represent RGB-color. '''

Vector4 = AlgTuple4
''' Alias for `AlgTuple4`. Represent four-dimensional vector (x, y, z, w).'''

BBox2 = Tuple[float, float, float, float]
''' Alias for tuple of four float value. Represent bounding box in two-dimensional measurement. '''

BBox3 = Tuple[float, float, float, float, float, float]
''' Alias for tuple of six float values. Represent bounding box in three-dimensional measurement. '''

Line2 = Tuple[float, float, float, float]
''' 
Alias for tuple of four float values. 
Represent line or segment passing through two points in two-dimensional measurement. 
'''

Line3 = Tuple[float, float, float, float, float, float]
''' 
Alias for tuple of six float values. 
Represent line or segment passing through two points in three-dimensional measurement.  
'''

Polygon2 = List[Point2]
'''
Alias for list of tuple[float, float] values.
Represent polygon in two-dimensional measurement.
'''

Mat2x2 = Tuple[Vector2, Vector2]
''' Alias. Represent Matrix with size 2x2. '''

Mat3x3 = Tuple[Vector3, Vector3, Vector3]
''' Alias. Represent matrix with size 3x3. '''


def is_strong_number(ob: Any) -> bool:
    return isinstance(ob, Number) and not isinstance(ob, bool)


def is_alg_tuple_n(ob: Any, n: int) -> bool:
    if not isinstance(ob, tuple):
        return False

    if len(ob) != n:
        return False

    for val in ob:
        if not is_strong_number(val):
            return False

    return True


def is_alg_tuple2(ob: Any) -> bool:
    return is_alg_tuple_n(ob, 2)


def is_alg_tuple3(ob: Any) -> bool:
    return is_alg_tuple_n(ob, 3)


def is_alg_tuple4(ob : Any) -> bool:
    return is_alg_tuple_n(ob, 4)


def is_point2(ob: Any) -> bool:
    return is_alg_tuple2(ob)


def is_vector2(ob: Any) -> bool:
    return is_alg_tuple2(ob)


def is_size2(ob: Any) -> bool:
    return is_alg_tuple2(ob)


def is_point3(ob: Any) -> bool:
    return is_alg_tuple3(ob)


def is_vector3(ob: Any) -> bool:
    return is_alg_tuple3(ob)


def is_size3(ob: Any) -> bool:
    return is_alg_tuple3(ob)


def is_color3(ob: Any) -> bool:
    return is_alg_tuple3(ob)


def is_bbox2(ob: Any) -> bool:
    return is_alg_tuple_n(ob, 4)


def is_bbox3(ob: Any) -> bool:
    return is_alg_tuple_n(ob, 6)


def is_line2(ob: Any) -> bool:
    return is_alg_tuple_n(ob, 4)


def is_line3(ob: Any) -> bool:
    return is_alg_tuple_n(ob, 6)


def is_polygon2(ob: Any) -> bool:
    if not isinstance(ob, list) and not isinstance(ob, tuple):
        return False

    if len(ob) < 3:
        return False

    for point in ob:
        if not is_point2(point):
            return False

    return True
