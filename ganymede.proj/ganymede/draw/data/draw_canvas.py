# python
from dataclasses import dataclass, field
# project
from ganymede.math.primitives import BBox2



@dataclass
class DrawCanvas:
    shapes     : list = field(default_factory=list)
    canvas_box : BBox2 = field(default_factory=lambda : (0.0, 0.0, 1.0, 1.0))