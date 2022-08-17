# python
from dataclasses import dataclass
from typing import Tuple
# project
import ganymede.random as g_random

from ganymede.augmentation.parameters import Rotate2dParameters


@dataclass
class Rotate2dDistribution:
    min_angle : float = -0.5
    max_angle : float =  0.5


    @staticmethod
    def load_from_dict(data : dict):
        return Rotate2dDistribution(
            data['min_angle'],
            data['max_angle']
        )


    def generate(self,random_instance) -> Rotate2dParameters:
        rs = random_instance

        angle = g_random.get_random_distance(self.min_angle, self.max_angle, rs)

        return Rotate2dParameters(angle)