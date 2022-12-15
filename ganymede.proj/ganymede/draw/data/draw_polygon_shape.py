# python
from dataclasses import dataclass
from enum import Enum, auto
# project
from ganymede.math.primitives import Polygon2, AlgTuple3



@dataclass
class DrawPolygonShape:
    polygon   : Polygon2
    color     : AlgTuple3
    thickness : float

