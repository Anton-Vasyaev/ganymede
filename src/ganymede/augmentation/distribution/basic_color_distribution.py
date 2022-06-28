from dataclasses import dataclass
from typing import Tuple


@dataclass
class BasicColorDistribution:
    red_values   : Tuple[float, float] = (0.5, 1.5)
    green_values : Tuple[float, float] = (0.5, 1.5)
    blue_values  : Tuple[float, float] = (0.5, 1.5)

    
    @staticmethod
    def load_from_dict(data):
        return BasicColorDistribution(
            data['red_values'],
            data['green_values'],
            data['blue_values']
        )