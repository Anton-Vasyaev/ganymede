# python
from dataclasses import dataclass
# project
from ganymede.math.primitives import BBox2, AlgTuple3


@dataclass
class DrawBBoxShape:
    bbox      : BBox2
    color     : AlgTuple3
    thickness : float