# python
import os
import xml.etree.ElementTree as ET
from typing import List, Any, cast
# 3rd party
from pathlib import Path

from .cvat_image_markup import CvatImageMarkup
# project
from .shapes import *


def parse_cvat_v1d1(
    xml_pathes     : List[str], 
    directory_path : str  = '.',
    exist_checking : bool = True
) -> List[List[CvatImageMarkup]]:
    directory_path_p = Path(directory_path)

    # checking xml exist
    for xml_path in xml_pathes:
        if not os.path.exists(xml_path): raise Exception(
            f'{xml_path} not exist'
        )

    task_list = []
    for xml_path in xml_pathes:
        root = ET.parse(xml_path).getroot()
        if root is None:
            raise Exception(f'Cannot parse xml:{xml_path}')

        task_name = cast(str, root.find('meta').find('task').find('name').text)

        data_list = []
        images = root.findall('image')
        for img in images:
            img_id   = int(cast(Any, img.get('id')))
            img_path = str(cast(Any, img.get('name')))
            img_w    = int(cast(Any, img.get('width')))
            img_h    = int(cast(Any, img.get('height')))

            img_path_p = directory_path_p / img_path

            polylines_el = img.findall('polyline')
            polylines = CvatPolyLineShape.load_from_xml_list(polylines_el, (img_w, img_h))

            polygons_el = img.findall('polygon')
            polygons = CvatPolygonShape.load_from_xml_list(polygons_el, (img_w, img_h))

            points_el = img.findall('points')
            points = CvatPointsShape.load_from_xml_list(points_el, (img_w, img_h))

            boxes_el = img.findall('box')
            boxes = CvatBoxShape.load_from_xml_list(boxes_el, (img_w, img_h))

            if exist_checking:
                if not img_path_p.exists():
                    raise Exception(
                        f'exist checking enable, path not exist:{img_path_p}'
                    )

            data_list.append(
                CvatImageMarkup(
                    str(img_path_p),
                    xml_path,
                    task_name,
                    img_id,
                    (img_w, img_h),
                    polygons,
                    polylines,
                    points,
                    boxes
                )
            )

        task_list.append(data_list)

    return task_list