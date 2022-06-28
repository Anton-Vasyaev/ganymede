# python
from dataclasses import dataclass, field
from typing import List
# project
from .shapes import *


@dataclass
class CvatImageMarkup:
    image_path  : str
    xml_path    : str
    task_name   : str
    image_id    : int
    image_size  : tuple
    polygons    : List[CvatPointsShape]
    polylines   : List[CvatPolyLineShape]
    points      : List[CvatPolyLineShape]
    boxes       : List[CvatBoxShape]
    meta_info   : dict = field(default_factory=dict)
    