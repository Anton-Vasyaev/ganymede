from dataclasses import dataclass
from .auxiliary import load_label_and_points_from_xml


@dataclass
class CvatPointsShape:
    label  : str
    points : list


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
        points_list = []
        for root in roots:
            points_list.append(CvatPointsShape.load_from_xml(root, normalize_size))

        return points_list
