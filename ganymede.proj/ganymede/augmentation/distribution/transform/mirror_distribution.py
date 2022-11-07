# python
from random import Random
from dataclasses import dataclass
# project
import ganymede.random as g_random
from ganymede.augmentation.distribution import IAugmentationDistribution
from ganymede.augmentation.parameters   import IAugmentationParameters
from  ganymede.augmentation.parameters import MirrorParameters


@dataclass
class MirrorDistribution(IAugmentationDistribution):
    horizontal : bool
    vertical   : bool


    def generate(self, generator: Random) -> IAugmentationParameters:
        gn = generator

        horizontal = g_random.get_random_bool(random_instance=gn) if self.horizontal else False
        vertical   = g_random.get_random_bool(random_instance=gn) if self.vertical   else False

        return MirrorParameters(horizontal, vertical)
