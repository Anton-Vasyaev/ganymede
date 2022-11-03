from dataclasses import dataclass
from typing import Tuple
# 3rd party
import ganymede.random as g_random
from ganymede.augmentation.parameters import BasicColorParameters


@dataclass
class BasicColorDistribution:
    red_values   : Tuple[float, float] = (0.5, 1.5)
    green_values : Tuple[float, float] = (0.5, 1.5)
    blue_values  : Tuple[float, float] = (0.5, 1.5)

    
    @staticmethod
    def load_from_dict(data):
        return BasicColorDistribution(
            data['red'],
            data['green'],
            data['blue']
        )


    def generate(
        self,
        random_instance
    ):
        rs = random_instance

        red   = g_random.get_random_distance(self.red_values[0],   self.red_values[1],   rs)
        green = g_random.get_random_distance(self.green_values[0], self.green_values[1], rs)
        blue  = g_random.get_random_distance(self.blue_values[0],  self.blue_values[1],  rs)

        return BasicColorParameters(red, green, blue)