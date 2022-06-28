# python
from dataclasses import dataclass
from typing      import Tuple


@dataclass
class PaddingDistribution:
    left_pads   : Tuple[float, float] = (-0.2, 0.2)
    right_pads  : Tuple[float, float] = (-0.2, 0.2)
    top_pads    : Tuple[float, float] = (-0.2, 0.2)
    bottom_pads : Tuple[float, float] = (-0.2, 0.2)


    @staticmethod
    def load_from_dict(data):
        return PaddingDistribution(
            data['left_pads'],
            data['right_pads'],
            data['top_pads'],
            data['bottom_pads']
        )