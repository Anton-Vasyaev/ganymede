from dataclasses import dataclass
from typing      import Tuple
from random      import Random
# project
import ganymede.random as g_random

from ganymede.augmentation.distribution import IAugmentationDistribution
from ganymede.augmentation.parameters import IAugmentationParameters
from ganymede.augmentation.parameters import Rotate3dParameters


@dataclass
class Rotate3dDistribution(IAugmentationDistribution):
    x_angles : Tuple[float, float]
    y_angles : Tuple[float, float] 
    z_angles : Tuple[float, float]


    def generate(self, generator: Random) -> IAugmentationParameters:
        gn = generator

        x_angle = g_random.get_random_distance(self.x_angles[0], self.x_angles[1], gn)
        y_angle = g_random.get_random_distance(self.y_angles[0], self.y_angles[1], gn)
        z_angle = g_random.get_random_distance(self.z_angles[0], self.z_angles[1], gn)

        return Rotate3dParameters(x_angle, y_angle, z_angle)