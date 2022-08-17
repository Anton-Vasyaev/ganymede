# project
import ganymede.math.bbox as m_bbox

from ganymede.math.bbox   import BBox
from ganymede.math.line2  import Line2D
from ganymede.math.point2 import Point2D
from ganymede.math.poly2  import Polygon2D



class SystemCoordTransformer:
    def __init__(
        self,
        src_coords : BBox,
        dst_coords : BBox
    ):

        s_l, s_t, s_r, s_b = src_coords
        d_l, d_t, d_r, d_b = dst_coords

        self.src_left   = s_l
        self.src_top    = s_t
        self.src_width  = m_bbox.width(src_coords)
        self.src_height = m_bbox.height(src_coords)

        self.dst_left   = d_l
        self.dst_top    = d_t
        self.dst_width  = m_bbox.width(dst_coords)
        self.dst_height = m_bbox.height(dst_coords)


    def transform_point(self, point : Point2D) -> Point2D:
        x, y = point

        x = (x - self.src_left) / self.src_width
        y = (y - self.src_top)  / self.src_height

        x = x * self.dst_width  + self.dst_left
        y = y * self.dst_height + self.dst_top

        return x, y


    def transform_line(self, line : Line2D) -> Line2D:
        p1, p2 = line
        tp1 = self.transform_point(p1)
        tp2 = self.transform_point(p2)

        return tp1, tp2


    def transform_polygon(self, poly : Polygon2D) -> Polygon2D:
        transform_poly = [None] * len(poly)
        for idx in range(len(poly)):
            poly_point      = poly[idx]
            transform_point = self.transform_point(poly_point)

            transform_poly[idx] = transform_point

        return transform_poly

    
    def transform_bbox(self, bbox : BBox) -> BBox:
        x1, y1, x2, y2 = bbox

        x1, y1 = self.transform_point((x1, y1))
        x2, y2 = self.transform_point((x2, y2))

        return x1, y1, x2, y2
