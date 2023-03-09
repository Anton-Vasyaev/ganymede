import dependencies
# python
import time
from dataclasses import dataclass
from typing import List, Tuple
from random import Random
# 3rd party
import numpy as np
# project
import ganymede.opencv as g_cv
import ganymede.json   as g_json

from ganymede.draw.text import draw_text_list

from ganymede.math.primitives import Point2

from ganymede.augmentation.distribution import *
from ganymede.augmentation.parameters   import *
from ganymede.augmentation import AugmentationData

from autofast.config import deserialize_config

@dataclass
class AugmentationExample:
    distribution : IAugmentationDistribution
    name         : str


@dataclass
class PointsData:
    tank_points : List[Point2]


def show_example(
    example     : AugmentationExample, 
    img         : np.ndarray, 
    points_data : PointsData,
    generator   : Random
):
    for idx in range(3):
        params = example.distribution.generate(generator)

        aug_data = AugmentationData(img, points_data.tank_points)

        aug_data = params.augmentate(aug_data)

        aug_img = aug_data.image
        points : List[Point2] = aug_data.points

        draw_text_list(aug_img, [(example.name, (0.05, 0.05), (255, 0, 0))])

        for p in points:
            g_cv.draw_circle(aug_img, p, (255, 0, 0), 2, 2)

        g_cv.imshow('debug', aug_img)


if __name__ == '__main__':
    generator = Random(time.time())

    img_path = r'../resources/images/tank.jpg'
    json_path = r'../resources/images/tank_data.json'

    img = g_cv.imread(img_path)

    points_dict = g_json.load_from_file(json_path)
    points_data = deserialize_config(PointsData, points_dict)

    distributions : List[IAugmentationDistribution] = [
        AugmentationExample(BasicColorDistribution((0.5, 1.5), (0.5, 1.5), (0.5, 1.5)), 'basic_color'),
        AugmentationExample(MirrorDistribution(True, True), 'mirror'),
        AugmentationExample(PaddingDistribution((-0.5, 0.2), (-0.3, 0.2), (-0.3, 0.2), (-0.3, 0.2)), 'padding'),
        AugmentationExample(Rotate2dDistribution(-45, 45), 'rotate2d'),
        AugmentationExample(StretchDistribution(-0.4, 0.4, None, None), 'stretch')
    ]

    for dist in distributions:
        show_example(dist, img, points_data, generator)