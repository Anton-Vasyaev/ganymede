# python
from dataclasses import dataclass
from typing      import List
# project
from .auxiliary import load_label_and_points_from_xml

from ganymede.math.point2 import Point2D

@dataclass
class CvatPolyLineShape:
    label  : str
    points : List[Point2D]


    @staticmethod
    def load_from_xml(
        root,
        normalize_size = (1.0, 1.0)
    ):
        label, points = load_label_and_points_from_xml(root, normalize_size)

        return CvatPolyLineShape(label, points)


    @staticmethod
    def load_from_xml_list(
        roots,
        normalize_size = (1.0, 1.0)
    ):
        polylines = []
        for root in roots:
            polylines.append(CvatPolyLineShape.load_from_xml(root, normalize_size))

        return polylines