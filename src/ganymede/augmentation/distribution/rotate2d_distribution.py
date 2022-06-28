# python
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Rotate2dDistribution:
    min_angle : float = -0.5
    max_angle : float =  0.5


    @staticmethod
    def load_from_dict(data):
        return Rotate2dDistribution(
            data['min_angle'],
            data['max_angle']
        )