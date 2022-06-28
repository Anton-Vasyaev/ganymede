# 3rd party
import cv2 as cv
import numpy as np
# project
import ganymede.math.point2 as g_p2



class PerspectiveCoordTransformer:
    @staticmethod
    def from_points(src_points, dst_points):
        src_p = np.float32(src_points)
        dst_p = np.float32(dst_points)

        mat = cv.getPerspectiveTransform(src_p, dst_p)
        
        return PerspectiveCoordTransformer(mat)
        
    
    def __init__(self, transform_mat):
        if not isinstance(transform_mat, list):
            transform_mat = transform_mat.tolist()
            
        self.transform_mat = transform_mat
        
        
    def __call__(self, coord):
        return g_p2.perspective_transform(coord, self.transform_mat)