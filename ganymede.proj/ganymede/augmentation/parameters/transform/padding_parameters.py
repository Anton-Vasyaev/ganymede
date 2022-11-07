# python
from dataclasses import dataclass
# 3rd party
import cv2 as cv # type: ignore
# project
from ganymede.augmentation.parameters.i_augmentation_parameters import IAugmentationParameters
from ganymede.augmentation.augmentation_data import AugmentationData
from ganymede.augmentation.function.transform.padding import augmentate_padding


@dataclass
class PaddingParameters(IAugmentationParameters):
    left_pad: float
    right_pad: float
    top_pad: float
    bottom_pad: float

    def augmentate(
        self,
        data: AugmentationData,
        interpolation: int = cv.INTER_AREA
    ) -> AugmentationData:
        return augmentate_padding(data, self.left_pad, self.right_pad, self.top_pad, self.bottom_pad)
