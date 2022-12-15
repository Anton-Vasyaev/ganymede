# python
from typing import List, Tuple
from xml.etree.ElementTree import ElementTree
# project
from ganymede.math.primitives import Point2, Size2


def load_label_and_points_from_xml(
    root : ElementTree,
    normalize_size : Size2 = (1.0, 1.0)
) -> Tuple[str, List[Point2]]:
    norm_w, norm_h = normalize_size

    label = root.get('label')
    points_str = root.get('points')

    points_str_l = points_str.split(';')
    points : List[Point2] = []
    for points_str in points_str_l:
        x, y = points_str.split(',')
        x, y = float(x) / norm_w, float(y) / norm_h

        points.append((x, y))

    return label, points