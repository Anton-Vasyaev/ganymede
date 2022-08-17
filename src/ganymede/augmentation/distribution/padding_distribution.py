# python
from dataclasses import dataclass
from typing      import Tuple
# project
import ganymede.random as g_random

from ganymede.augmentation.parameters import PaddingParameters


@dataclass
class PaddingDistribution:
    left_pads   : Tuple[float, float] = (-0.2, 0.2)
    right_pads  : Tuple[float, float] = (-0.2, 0.2)
    top_pads    : Tuple[float, float] = (-0.2, 0.2)
    bottom_pads : Tuple[float, float] = (-0.2, 0.2)


    @staticmethod
    def load_from_dict(data):
        return PaddingDistribution(
            data['left'],
            data['right'],
            data['top'],
            data['bottom']
        )

    def generate(self, random_instance = None) -> PaddingParameters:
        rs = random_instance

        left  = g_random.get_random_distance(self.left_pads[0],  self.left_pads[1], rs)
        right = g_random.get_random_distance(self.right_pads[0], self.right_pads[1], rs)

        top    = g_random.get_random_distance(self.top_pads[0], self.top_pads[1], rs)
        bottom = g_random.get_random_distance(self.bottom_pads[0], self.top_pads[1], rs)

        return PaddingParameters(left, right, top, bottom)