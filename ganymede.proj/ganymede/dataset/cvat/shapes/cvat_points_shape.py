# python
from dataclasses import dataclass
from typing      import List
# project
from .auxiliary import load_label_and_points_from_xml
from ganymede.math.primitives import Point2


@dataclass
class CvatPointsShape:
    label  : str
    points : List[Point2]


    @staticmethod
    def load_from_xml(
        root,
        normalize_size = (1.0, 1.0)
    ):
        label, points = load_label_and_points_from_xml(root, normalize_size)

        return CvatPointsShape(label, points)


    @staticmethod
    def load_from_xml_list(
        roots,
        normalize_size = (1.0, 1.0)
    ):
        points_list : List[CvatPointsShape] = []
        for root in roots:
            points_list.append(CvatPointsShape.load_from_xml(root, normalize_size))

        return points_list
