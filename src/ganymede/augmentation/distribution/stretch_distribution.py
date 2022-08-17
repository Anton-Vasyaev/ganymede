# python
from dataclasses import dataclass
# project
import ganymede.random as g_random

from ganymede.augmentation.parameters import StretchType, StretchOrientation, StretchParameters


@dataclass
class StretchDistribution:
    min_offset  : float = -0.3
    max_offset  : float =  0.3
    type        : StretchType = None
    orientation : StretchOrientation = None

    
    @staticmethod
    def load_from_dict(data):
        return StretchDistribution(
            data['min_offset'],
            data['max_offset'],
            None if data['type'] is None else StretchType.from_str(data['type']),
            None if data['orientation'] is None else StretchType.from_str(data['orientation'])
        )


    def generate(
        self,
        random_instance
    ):
        rs = random_instance

        offset = g_random.get_random_distance(self.min_offset, self.max_offset)

        s_type = self.type
        if s_type is None:
            s_type = g_random.get_random_enum(StretchType, rs)

        orientation = self.orientation
        if orientation is None:
            orientation = g_random.get_random_enum(StretchOrientation, rs)

        return StretchParameters(
            offset,
            s_type,
            orientation
        )