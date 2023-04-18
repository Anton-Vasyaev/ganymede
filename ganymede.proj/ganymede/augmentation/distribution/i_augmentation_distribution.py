# python
from dataclasses import dataclass
from random import Random
from abc    import abstractmethod
# project

from ganymede.augmentation.parameters.i_augmentation_parameters import IAugmentationParameters


class IAugmentationDistribution:
    @abstractmethod
    def generate(
        self, 
        generator : Random
    ) -> IAugmentationParameters:
        raise NotImplementedError()