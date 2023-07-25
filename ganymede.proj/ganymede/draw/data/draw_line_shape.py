# python
from dataclasses import dataclass
# project
from .canvas_shape import CanvasShape
from ganymede.math.primitives import Line2, AlgTuple3


@dataclass
class DrawLineShape(CanvasShape):
    line      : Line2
    color     : AlgTuple3
    thickness : float