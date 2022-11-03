# python
from dataclasses import dataclass


@dataclass
class Rotate3dParameters:
    x_angle : float
    y_angle : float
    z_angle : float