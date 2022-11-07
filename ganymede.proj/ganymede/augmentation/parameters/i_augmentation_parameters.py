# python
from abc import abstractmethod
# 3rd party
import cv2 as cv
# project
from ganymede.augmentation.augmentation_data import AugmentationData


class IAugmentationParameters:
    @abstractmethod
    def augmentate(
        self,
        data: AugmentationData,
        interpolation: int = cv.INTER_AREA
    ) -> AugmentationData:
        raise NotImplementedError()
