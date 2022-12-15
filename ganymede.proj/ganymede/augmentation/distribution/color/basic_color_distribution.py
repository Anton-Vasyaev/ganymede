# python
from random import Random
from dataclasses import dataclass
from typing import Tuple
# 3rd party
import ganymede.random as g_random
from ganymede.augmentation.parameters.color.basic_color_parameters import BasicColorParameters
from ganymede.augmentation.distribution.i_augmentation_distribution import IAugmentationDistribution
from ganymede.augmentation.parameters.i_augmentation_parameters import IAugmentationParameters

@dataclass
class BasicColorDistribution(IAugmentationDistribution):
    red_values   : Tuple[float, float]
    green_values : Tuple[float, float]
    blue_values  : Tuple[float, float]

    def generate(self, generator: Random) -> IAugmentationParameters:
        gn = generator

        red   = g_random.get_random_distance(self.red_values[0],   self.red_values[1],   gn)
        green = g_random.get_random_distance(self.green_values[0], self.green_values[1], gn)
        blue  = g_random.get_random_distance(self.blue_values[0],  self.blue_values[1],  gn)

        return BasicColorParameters(red, green, blue)