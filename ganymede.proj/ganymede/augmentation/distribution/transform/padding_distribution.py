# python
from dataclasses import dataclass
from typing      import Tuple
from random      import Random
# project
import ganymede.random as g_random

from ganymede.augmentation.parameters import IAugmentationParameters
from ganymede.augmentation.distribution import IAugmentationDistribution
from ganymede.augmentation.parameters import PaddingParameters



@dataclass
class PaddingDistribution(IAugmentationDistribution):
    left_pads   : Tuple[float, float]
    right_pads  : Tuple[float, float]
    top_pads    : Tuple[float, float] 
    bottom_pads : Tuple[float, float]

    def generate(self, generator: Random) -> IAugmentationParameters:
        gn = generator

        left  = g_random.get_random_distance(self.left_pads[0],  self.left_pads[1], gn)
        right = g_random.get_random_distance(self.right_pads[0], self.right_pads[1], gn)

        top    = g_random.get_random_distance(self.top_pads[0], self.top_pads[1], gn)
        bottom = g_random.get_random_distance(self.bottom_pads[0], self.top_pads[1], gn)

        return PaddingParameters(left, right, top, bottom)