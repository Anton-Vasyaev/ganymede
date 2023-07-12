# python
from dataclasses import dataclass
# project
from ganymede.draw.data import *


@dataclass
class ImageProcessingDrawInfo:
    debug_name : str
    
    frame_name : str

    canvas : DrawCanvas

