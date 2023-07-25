# python
from dataclasses import dataclass
from enum import Enum, auto
# project
from .canvas_shape import CanvasShape
from ganymede.math.primitives import Polygon2, AlgTuple3


@dataclass
class DrawPolygonShape(CanvasShape):
    polygon   : Polygon2
    color     : AlgTuple3
    thickness : float

