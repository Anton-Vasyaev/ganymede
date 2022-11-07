# python
from abc import abstractmethod
from dataclasses import dataclass
# 3rd party
import cv2 as cv # type: ignore
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
