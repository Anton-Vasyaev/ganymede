# python
from dataclasses import dataclass
# project
from .canvas_shape import CanvasShape
from ganymede.math.primitives import Point2, AlgTuple3


@dataclass
class DrawPointShape(CanvasShape):
    point  : Point2
    color  : AlgTuple3
    radius : float