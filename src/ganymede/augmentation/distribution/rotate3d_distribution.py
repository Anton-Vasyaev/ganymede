from dataclasses import dataclass
from typing      import Tuple


@dataclass
class Rotate3dDistribution:
    x_angles : Tuple[float, float] = (-0.5, 0.5)
    y_angles : Tuple[float, float] = (-0.5, 0.5)
    z_angles : Tuple[float, float] = (-0.5, 0.5)


    @staticmethod
    def load_from_dict(data):
        return Rotate3dDistribution(
            data['x_angles'],
            data['y_angles'],
            data['z_angles']
        )