# python
from dataclasses import dataclass, field
from typing      import List
# project
from .draw_bbox_shape    import DrawBBoxShape
from .draw_line_shape    import DrawLineShape
from .draw_point_shape   import DrawPointShape
from .draw_polygon_shape import DrawPolygonShape
from .fill_polygon_shape import FillPolygonShape

from ganymede.math.primitives import BBox2



@dataclass
class DrawCanvas:
    draw_boxes : List[DrawBBoxShape] = field(default_factory=list)

    draw_lines : List[DrawLineShape] = field(default_factory=list)

    draw_points : List[DrawPointShape] = field(default_factory=list)

    draw_polygons : List[DrawPolygonShape] = field(default_factory=list)

    fill_polygons : List[FillPolygonShape] = field(default_factory=list)

    canvas_box : BBox2 = field(default_factory=lambda : (0.0, 0.0, 1.0, 1.0))