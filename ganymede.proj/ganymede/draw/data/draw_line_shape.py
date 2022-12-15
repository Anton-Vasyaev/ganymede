# python
from dataclasses import dataclass
# project
from ganymede.math.primitives import Line2, AlgTuple3


@dataclass
class DrawLineShape:
    line      : Line2
    color     : AlgTuple3
    thickness : float