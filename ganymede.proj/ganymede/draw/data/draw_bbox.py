# python
from dataclasses import dataclass
# project
from ganymede.math.alg_tuple3 import AlgTuple3
from ganymede.math.bbox       import BBox


@dataclass
class DrawBBox:
    bbox      : BBox
    color     : AlgTuple3
    thickness : float