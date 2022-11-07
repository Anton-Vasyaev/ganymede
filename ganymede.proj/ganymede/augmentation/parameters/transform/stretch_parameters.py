# python
from dataclasses import dataclass
# 3rd party
import cv2 as cv
# project
from ganymede.augmentation.parameters.transform.stretch_orientation import StretchOrientation
from ganymede.augmentation.parameters.transform.stretch_type import StretchType
from ganymede.augmentation.parameters import IAugmentationParameters
from ganymede.augmentation import AugmentationData
from ganymede.augmentation.transformation.stretch import augmentate_stretch


@dataclass
class StretchParameters(IAugmentationParameters):
    offset: float
    orientation: StretchOrientation
    stretch_type: StretchType

    def augmentate(
        self,
        data: AugmentationData,
        interpolation: int = cv.INTER_AREA
    ) -> AugmentationData:
        return augmentate_stretch(data, self.offset, self.orientation, self.stretch_type, interpolation)
