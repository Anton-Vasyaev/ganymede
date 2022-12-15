# python
from dataclasses import dataclass
# 3rd party
import cv2 as cv # type: ignore
# project
from ganymede.augmentation.parameters.i_augmentation_parameters import IAugmentationParameters
from ganymede.augmentation.augmentation_data import AugmentationData
from ganymede.augmentation.function.color.basic_color import augmentate_basic_color

@dataclass
class BasicColorParameters(IAugmentationParameters):
    red   : float
    green : float
    blue  : float

    def augmentate(self, data: AugmentationData, interpolation: int = cv.INTER_AREA) -> AugmentationData:
        aug_image = augmentate_basic_color(data.image, self.red, self.green, self.blue)

        return AugmentationData(aug_image, data.points)