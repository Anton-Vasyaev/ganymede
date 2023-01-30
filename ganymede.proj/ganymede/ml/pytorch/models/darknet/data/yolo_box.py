# pytohn
from typing      import List
from dataclasses import dataclass
# 3rd party
from ganymede.math.primitives import BBox2


@dataclass
class YoloBox:
    bbox : BBox2

    obj_conf : float

    class_probs : List[float]