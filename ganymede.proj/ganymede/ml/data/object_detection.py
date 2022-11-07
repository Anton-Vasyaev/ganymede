# python
from dataclasses import dataclass
# project
from ganymede.math.primitives import BBox2


@dataclass
class ObjectDetection:
    bbox                : BBox2
    class_id            : int
    object_confidence   : float
    class_confidence    : float