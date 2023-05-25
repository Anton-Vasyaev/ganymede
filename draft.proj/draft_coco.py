import dependencies
# python
import os
import time
from typing      import List, cast
from dataclasses import dataclass, field
from numbers     import Number
# 3rd party
import numpy as np

from autofast.config import *
# project
import ganymede.json   as g_js
import ganymede.opencv as g_cv

from ganymede.math.primitives import Polygon2, BBox2

from ganymede.dataset.coco.coco_dataset_loader import CocoDatasetLoader

def draft_coco():
    markup_path = r'D:\datasets\coco\annotations\instances_val2014.json'

    images_dir = r'D:\datasets\coco\val2014'

    allowed_classes = [
        'person',
        'car',
        'motorcycle',
        'bus',
        'truck'
    ]

    print(f'loading...')
    loader = CocoDatasetLoader(markup_path, images_dir, allowed_classes)
    print(f'Done!')

    print(f'class names:')
    for _, class_name in loader.get_class_names().items():
        print(class_name)
    print()

    print(f'Images Counts:{len(loader)}.')

    for idx in range(len(loader)):
        image, objects = loader[idx]

        show_img = g_cv.resize_frame(image, (1600, 900))

        for obj in objects:
            for segment in obj.segments:
                g_cv.draw_polygon(show_img, segment, (0, 255, 0), 2)

        g_cv.imshow('debug', show_img)


if __name__ == '__main__':
    draft_coco()