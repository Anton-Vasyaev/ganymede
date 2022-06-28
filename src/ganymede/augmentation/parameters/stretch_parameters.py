# python
from dataclasses import dataclass
# project
from .stretch_orientation import StretchOrientation
from .stretch_type        import StretchType


@dataclass
class StretchParameters:
    offset      : float
    orientation : StretchOrientation
    type        : StretchType