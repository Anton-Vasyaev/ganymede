# python
from dataclasses import dataclass
from typing      import List
# project
from ganymede.math.primitives import Polygon2


@dataclass
class SegmentObjectMarkup:
    segments : List[Polygon2]

    class_id : int