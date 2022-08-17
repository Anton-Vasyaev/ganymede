# python
from dataclasses import dataclass
from enum import Enum, auto
# project
from ganymede.math.alg_tuple3 import AlgTuple3
from ganymede.math.poly2      import Polygon2D




@dataclass
class DrawPolygon:
    polygon   : Polygon2D
    color     : AlgTuple3
    thickness : float

