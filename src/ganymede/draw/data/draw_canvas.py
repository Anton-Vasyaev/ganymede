# python
from dataclasses import dataclass, field
# project
from ganymede.math.bbox import BBox
from .draw_shapes import DrawShapes


@dataclass
class DrawCanvas:
    shapes     : list = field(default_factory=list)
    canvas_box : BBox = field(default_factory=lambda : [0.0, 0.0, 1.0, 1.0])