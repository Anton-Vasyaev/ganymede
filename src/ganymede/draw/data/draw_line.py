# python
from dataclasses import dataclass
# project
from ganymede.math.alg_tuple3 import AlgTuple3
from ganymede.math.line2      import Line2D


@dataclass
class DrawLine:
    line      : Line2D
    color     : AlgTuple3
    thickness : float