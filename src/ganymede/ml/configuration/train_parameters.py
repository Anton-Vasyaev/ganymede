# python
from dataclasses import dataclass
from typing import Tuple
# project
from ganymede.imaging import ImageType


@dataclass
class TrainParameters:
    learning_rate : float
    epochs : int
    enable_gpu : bool
    input_size : Tuple[int, int]
    input_type : ImageType
    batch_size : int

    @staticmethod
    def load_from_dict(data):
        return TrainParameters(
            data['learning_rate'],
            data['epochs'],
            data['enable_gpu'],
            data['input_size'],
            ImageType.from_str(data['input_type']),
            data['batch_size']
        )