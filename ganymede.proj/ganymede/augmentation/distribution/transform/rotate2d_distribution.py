# python
from dataclasses import dataclass
from typing import Tuple
from random import Random
# project
import ganymede.random as g_random

from ganymede.augmentation.distribution import IAugmentationDistribution
from ganymede.augmentation.parameters   import IAugmentationParameters
from ganymede.augmentation.parameters import Rotate2dParameters



@dataclass
class Rotate2dDistribution(IAugmentationDistribution):
    min_angle : float

    max_angle : float

    def generate(self, generator: Random) -> IAugmentationParameters:
        gn = generator

        angle = g_random.get_random_distance(self.min_angle, self.max_angle, gn)

        return Rotate2dParameters(angle)