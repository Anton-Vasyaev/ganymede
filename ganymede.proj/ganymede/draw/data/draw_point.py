# python
from dataclasses import dataclass
# project
from ganymede.math.alg_tuple3 import AlgTuple3
from ganymede.math.point2     import Point2D


@dataclass
class DrawPoint:
    point  : Point2D
    color  : AlgTuple3
    radius : float