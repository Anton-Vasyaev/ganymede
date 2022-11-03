# python
from typing import List
from dataclasses import dataclass, field
# project
from .draw_bbox    import DrawBBox
from .draw_line    import DrawLine
from .draw_point   import DrawPoint
from .draw_polygon import DrawPolygon
from .fill_polygon import FillPolygon


@dataclass
class DrawShapes:
    draw_lines      : List[DrawLine]    = field(default_factory=list)
    draw_points     : List[DrawPoint]   = field(default_factory=list)
    draw_polygons   : List[DrawPolygon] = field(default_factory=list)
    draw_bboxes     : List[DrawBBox]    = field(default_factory=list)

    fill_polygons : List[FillPolygon] = field(default_factory=list)