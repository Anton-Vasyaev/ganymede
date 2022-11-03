# python
from dataclasses import dataclass
# project
from ganymede.math.bbox import BBox


@dataclass
class BBoxObjectMarkup:
    bbox     : BBox
    class_id : int