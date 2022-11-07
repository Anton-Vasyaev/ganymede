# python
from dataclasses import dataclass
# 3rd party
import cv2 as cv # type: ignore
# project
from ganymede.augmentation.parameters.i_augmentation_parameters import IAugmentationParameters
from ganymede.augmentation.augmentation_data import AugmentationData


@dataclass
class Rotate3dParameters(IAugmentationParameters):
    x_angle : float
    y_angle : float
    z_angle : float

    def augmentate(self, data: AugmentationData, interpolation: int = cv.INTER_AREA) -> AugmentationData:
        raise NotImplementedError()