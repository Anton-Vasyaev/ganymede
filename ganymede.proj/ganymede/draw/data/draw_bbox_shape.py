# python
from dataclasses import dataclass
# project
from .canvas_shape import CanvasShape
from ganymede.math.primitives import BBox2, AlgTuple3


@dataclass
class DrawBBoxShape(CanvasShape):
    bbox      : BBox2
    color     : AlgTuple3
    thickness : float