# python
from dataclasses import dataclass
# project
from ganymede.math.bbox import BBox


@dataclass
class CvatBoxShape:
    label : str
    box   : BBox


    @staticmethod
    def load_from_xml(root, normalize_size):
        norm_w, norm_h = normalize_size

        label = root.get('label')

        xtl   = float(root.get('xtl')) / norm_w
        ytl   = float(root.get('ytl')) / norm_h
        xbr   = float(root.get('xbr')) / norm_w
        ybr   = float(root.get('ybr')) / norm_h

        return CvatBoxShape(label, (xtl, ytl, xbr, ybr))


    @staticmethod
    def load_from_xml_list(
        roots,
        normalize_size = (1.0, 1.0)
    ):
        boxes_list = []
        for root in roots:
            boxes_list.append(CvatBoxShape.load_from_xml(root, normalize_size))

        return boxes_list