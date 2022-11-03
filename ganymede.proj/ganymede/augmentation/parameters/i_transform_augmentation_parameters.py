# python
from abc import ABC, abstractmethod
# project
from ganymede.augmentation.augmentation_data import AugmentationData


class ITransformAugmentationParameters(ABC):
    @abstractmethod
    def transform_augmentate(
        self, 
        data : AugmentationData
    ):
        raise NotImplementedError()