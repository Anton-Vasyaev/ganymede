# python
from dataclasses import dataclass
# 3rd party
import cv2 as cv
# project
from ganymede.augmentation.parameters.i_augmentation_parameters import IAugmentationParameters
from ganymede.augmentation.augmentation_data import AugmentationData
from ganymede.augmentation.transformation.mirror import augmentate_mirror


@dataclass
class MirrorParameters(IAugmentationParameters):
    horizontal: bool

    vertical: bool

    def augmentate(
        self,
        data: AugmentationData,
        interpolation: int = cv.INTER_AREA
    ) -> AugmentationData:
        return augmentate_mirror(data, self.horizontal, self.vertical)
