# python
from dataclasses import dataclass
from multiprocessing.reduction import steal_handle
# project
from ganymede.augmentation.parameters import StretchType, StretchOrientation

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