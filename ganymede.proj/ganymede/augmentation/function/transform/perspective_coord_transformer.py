# python
# 3rd party
# project
import ganymede.opencv as g_cv
import ganymede.math.point2 as g_p2
from ganymede.math.primitives import Mat3x3
from ganymede.opencv.special import PerspectiveType


class PerspectiveCoordTransformer:
    transform_mat: Mat3x3

    @staticmethod
    def from_points(src_points: PerspectiveType, dst_points: PerspectiveType):
        mat = g_cv.get_perspective_transform(src_points, dst_points)

        return PerspectiveCoordTransformer(mat)

    def __init__(self, transform_mat: Mat3x3):
        self.transform_mat = transform_mat

    def __call__(self, coord):
        return g_p2.perspective_transform(coord, self.transform_mat)
