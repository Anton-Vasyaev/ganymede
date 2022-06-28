# python
from dataclasses import dataclass
# project
from .auxiliary import load_label_and_points_from_xml

@dataclass
class CvatPolygonShape:
    label : str
    points : str
    
    @staticmethod
    def load_from_xml(
        root,
        normalize_size = (1.0, 1.0)
    ):
        label, points = load_label_and_points_from_xml(root, normalize_size)

        return CvatPolygonShape(label, points)


    @staticmethod
    def load_from_xml_list(
        roots,
        normalize_size = (1.0, 1.0)
    ):
        polygons = []
        for root in roots:
            polygons.append(CvatPolygonShape.load_from_xml(root, normalize_size))

        return polygons

