# 3rd party
import cv2 as cv
# project
import ganymede.opencv as g_cv
from ganymede.augmentation.augmentation_data import AugmentationData
from .perspective_coord_transformer          import PerspectiveCoordTransformer
from .delegate                               import delegate_transform_points



def augmentation_perspective(
    data : AugmentationData,
    src_points : list,
    dst_points : list,
    interpolation = cv.INTER_AREA
) -> AugmentationData:
    img = g_cv.warp_perspective_keypoints(data.image, src_points, dst_points, interpolation)
    
    transform = PerspectiveCoordTransformer.from_points(src_points, dst_points)
    
    points = delegate_transform_points(data.points, transform)
    
    return AugmentationData(img, points)