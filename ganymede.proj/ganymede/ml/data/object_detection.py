# python
from dataclasses import dataclass
# project
from ganymede.math.bbox import BBox


@dataclass
class ObjectDetection:
    bbox                : BBox
    class_id            : int
    object_confidence   : float
    class_confidence    : float