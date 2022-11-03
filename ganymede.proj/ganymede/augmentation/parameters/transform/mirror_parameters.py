# python
from dataclasses import dataclass


@dataclass
class MirrorParameters:
    horizontal : bool
    vertical   : bool