# 3rd party
import numpy as np
# project
from ganymede.augmentation.augmentation_data import AugmentationData
from .delegate                               import delegate_transform_points


class PaddingCoordTransformer:
    def __init__(
        self,
        left_pad, 
        right_pad, 
        top_pad, 
        bottom_pad
    ):
        new_w = 1.0 + right_pad  - (0.0 - left_pad)
        new_h = 1.0 + bottom_pad - (0.0 - top_pad)
        
        self.w_scale = 1.0 / new_w
        self.h_scale = 1.0 / new_h
        
        self.x_offset = left_pad / new_w
        self.y_offset = top_pad  / new_h
        
    
    def __call__(self, coord):
        x, y = coord
        
        x = self.x_offset + x * self.w_scale
        y = self.y_offset + y * self.h_scale
        
        return x, y



def __augmentation_padding_img(
    img        : np.ndarray, 
    left_pad   : float, 
    right_pad  : float, 
    top_pad    : float, 
    bottom_pad : float
) -> np.ndarray:
    img_h, img_w = img.shape[0:2]
    channels = None
    if len(img.shape) == 3: channels = img.shape[2]

    left_abs   = int(img_w * left_pad)
    right_abs  = int(img_w * right_pad)
    top_abs    = int(img_h * top_pad)
    bottom_abs = int(img_h * bottom_pad)

    new_w = (img_w + left_abs + right_abs)
    new_h = (img_h + top_abs  + bottom_abs)

    new_img_shape = (new_h, new_w)
    if not channels is None: new_img_shape += (channels,)
    new_img = np.zeros(new_img_shape, dtype=np.uint8)

    # get roi of source image
    sx1 = 0
    sy1 = 0
    sx2 = img_w
    sy2 = img_h

    if left_abs < 0:   sx1 -= left_abs
    if right_abs < 0:  sx2 += right_abs
    if top_abs < 0:    sy1 -= top_abs
    if bottom_abs < 0: sy2 += bottom_abs

    # get roi of dst image
    dx1 = 0
    dy1 = 0
    dx2 = new_w
    dy2 = new_h

    #print(f'new size:{new_w, new_h}')

    if left_abs > 0:   dx1 += left_abs
    if right_abs > 0:  dx2 -= right_abs
    if top_abs > 0:    dy1 += top_abs
    if bottom_abs > 0: dy2 -= bottom_abs
    
    new_img[dy1:dy2, dx1:dx2] = img[sy1:sy2, sx1:sx2]

    return new_img


def augmentation_padding(
    data       : AugmentationData,
    left_pad   : float,
    right_pad  : float,
    top_pad    : float,
    bottom_pad : float
) -> AugmentationData:
    image = __augmentation_padding_img(
        data.image, 
        left_pad, 
        right_pad, 
        top_pad, 
        bottom_pad
    )

    transformer = PaddingCoordTransformer(left_pad, right_pad, top_pad, bottom_pad)
    
    points = delegate_transform_points(data.points, transformer)
    
    return AugmentationData(image, points)