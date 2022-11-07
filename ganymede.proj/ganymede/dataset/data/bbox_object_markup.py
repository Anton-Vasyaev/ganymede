# python
from dataclasses import dataclass
# project
from ganymede.math.primitives import BBox2


@dataclass
class BBoxObjectMarkup:
    bbox     : BBox2
    class_id : int