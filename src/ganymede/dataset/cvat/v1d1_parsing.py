# python
import os
import xml.etree.ElementTree as ET
from typing import List
# 3rd party
import cv2 as cv
from pathlib import Path

from .cvat_image_markup import CvatImageMarkup
# project
from .shapes import *


def parse_cvat_v1d1(
    xml_pathes     : List[str], 
    directory_path : str  = '.',
    exist_checking : bool = True
) -> List[CvatImageMarkup]:
    directory_path_p = Path(directory_path)

    # checking xml exist
    for xml_path in xml_pathes:
        if not os.path.exists(xml_path): raise Exception(
            f'{xml_path} not exist'
        )

    task_list = []
    for xml_path in xml_pathes:
        root = ET.parse(xml_path).getroot()

        task_name = root.find('meta').find('task').find('name').text

        data_list = []
        images = root.findall('image')
        for img in images:
            img_id   = int(img.get('id'))
            img_path = img.get('name')
            img_w    = int(img.get('width'))
            img_h    = int(img.get('height'))

            img_path_p = directory_path_p / img_path

            polylines = img.findall('polyline')
            polylines = CvatPolyLineShape.load_from_xml_list(polylines, (img_w, img_h))

            polygons = img.findall('polygon')
            polygons = CvatPolygonShape.load_from_xml_list(polygons, (img_w, img_h))

            points = img.findall('points')
            points = CvatPointsShape.load_from_xml_list(points, (img_w, img_h))

            boxes = img.findall('box')
            boxes = CvatBoxShape.load_from_xml_list(boxes, (img_w, img_h))

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