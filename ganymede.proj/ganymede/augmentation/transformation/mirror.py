# 3rd party
import cv2 as cv
# project
import ganymede.math.point2 as point2
from ganymede.augmentation.augmentation_data import AugmentationData
from .delegate                               import delegate_transform_points



class MirrorCoordTransformer:
    def __init__(self, horizontal, vertical):
        self.horizontal = horizontal
        self.vertical   = vertical
        
        
    def __call__(self, coord):
        x, y = coord
        
        if self.horizontal: x = 1.0 - x
        if self.vertical:   y = 1.0 - y
        
        return x, y


def __augmentation_mirror_img(
    img,
    horizontal,
    vertical
):
    flip_value = None

    if not horizontal and not vertical: return img.copy()
    elif horizontal and vertical: flip_value = -1
    elif horizontal:              flip_value =  1
    elif vertical:                flip_value =  0

    img = cv.flip(img, flip_value)

    return img


def augmentation_mirror(
    data       : AugmentationData,
    horizontal : bool = True,
    vertical   : bool = True
) -> AugmentationData:
    
    image = __augmentation_mirror_img(data.image, horizontal, vertical)
    
    transformer = MirrorCoordTransformer(horizontal, vertical)
    
    points = delegate_transform_points(data.points, transformer)
    
    return AugmentationData(image, points)