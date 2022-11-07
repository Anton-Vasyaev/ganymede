# project
import ganymede.math.bbox2 as m_bbox2
import ganymede.math.line2 as m_line2
from ganymede.math.primitives import BBox2, Point2, Line2, Polygon2


class SystemCoordTransformer:
    def __init__(
        self,
        src_coords: BBox2,
        dst_coords: BBox2
    ):

        s_l, s_t, _, _ = src_coords
        d_l, d_t, _, _ = dst_coords

        self.src_left = s_l
        self.src_top = s_t
        self.src_width = m_bbox2.width(src_coords)
        self.src_height = m_bbox2.height(src_coords)

        self.dst_left = d_l
        self.dst_top = d_t
        self.dst_width = m_bbox2.width(dst_coords)
        self.dst_height = m_bbox2.height(dst_coords)

    def transform_point(self, point: Point2) -> Point2:
        x, y = point

        x = (x - self.src_left) / self.src_width
        y = (y - self.src_top) / self.src_height

        x = x * self.dst_width + self.dst_left
        y = y * self.dst_height + self.dst_top

        return x, y

    def transform_line(self, line: Line2) -> Line2:
        p1 = m_line2.first(line)
        p2 = m_line2.second(line)

        tp1 = self.transform_point(p1)
        tp2 = self.transform_point(p2)

        return m_line2.from_points(tp1, tp2)

    def transform_polygon(self, poly: Polygon2) -> Polygon2:
        transform_poly: Polygon2 = [Point2()] * len(poly)
        for idx in range(len(poly)):
            poly_point = poly[idx]
            transform_point = self.transform_point(poly_point)

            transform_poly[idx] = transform_point

        return transform_poly

    def transform_bbox(self, bbox: BBox2) -> BBox2:
        x1, y1, x2, y2 = bbox

        x1, y1 = self.transform_point((x1, y1))
        x2, y2 = self.transform_point((x2, y2))

        return x1, y1, x2, y2
