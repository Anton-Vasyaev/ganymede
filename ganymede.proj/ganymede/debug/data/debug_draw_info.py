# python
from dataclasses import dataclass, field
from typing      import Optional
# project
from ganymede.draw.data import *


@dataclass
class DebugDrawInfo:
    debug_name : str
    
    image_id : Optional[str] = field(default=None)

    canvas : DrawCanvas = field(default_factory=DrawCanvas)

