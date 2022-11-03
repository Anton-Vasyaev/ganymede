# python
from dataclasses import dataclass


@dataclass
class PaddingParameters:
    left   : float
    right  : float
    top    : float
    bottom : float