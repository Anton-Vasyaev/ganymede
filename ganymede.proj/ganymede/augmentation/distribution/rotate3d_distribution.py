from dataclasses import dataclass
from typing      import Tuple
# project
import ganymede.random as g_random

from ganymede.augmentation.parameters import Rotate3dParameters


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

    
    def generate(
        self,
        random_instance
    ):
        rs = random_instance

        x_angle = g_random.get_random_distance(self.x_angles[0], self.x_angles[1], rs)
        y_angle = g_random.get_random_distance(self.y_angles[0], self.y_angles[1], rs)
        z_angle = g_random.get_random_distance(self.z_angles[0], self.z_angles[1], rs)

        return Rotate3dParameters(x_angle, y_angle, z_angle)