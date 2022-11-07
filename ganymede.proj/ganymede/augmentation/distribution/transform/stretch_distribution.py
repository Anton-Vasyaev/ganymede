# python
from random import Random
from dataclasses import dataclass
from typing import Optional
# project
import ganymede.random as g_random

from ganymede.augmentation.parameters import IAugmentationParameters
from ganymede.augmentation.distribution import IAugmentationDistribution
from ganymede.augmentation.parameters import StretchType, StretchOrientation, StretchParameters



@dataclass
class StretchDistribution(IAugmentationDistribution):
    min_offset : float

    max_offset : float

    stretch_type : StretchType

    orientation : StretchOrientation


    def generate(self, generator: Random) -> IAugmentationParameters:
        gn = generator

        offset = g_random.get_random_distance(self.min_offset, self.max_offset, gn)

        s_type = self.stretch_type
        if s_type is None:
            s_type = g_random.get_random_enum(StretchType, gn)

        s_orientation = self.orientation
        if s_orientation is None:
            s_orientation = g_random.get_random_enum(StretchOrientation, gn)

        return StretchParameters(
            offset,
            s_orientation,
            s_type
        )