# python
from dataclasses import dataclass
# 3rd party
import cv2 as cv # type: ignore
# project
from ganymede.augmentation.function.transform.stretch_data.stretch_orientation import StretchOrientation
from ganymede.augmentation.function.transform.stretch_data.stretch_type import StretchType
from ganymede.augmentation.parameters.i_augmentation_parameters import IAugmentationParameters
from ganymede.augmentation.augmentation_data import AugmentationData
from ganymede.augmentation.function.transform.stretch import augmentate_stretch


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
