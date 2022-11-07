# python
from dataclasses import dataclass
# project
from ganymede.math.primitives import Point2, AlgTuple3


@dataclass
class DrawPointShape:
    point  : Point2
    color  : AlgTuple3
    radius : float