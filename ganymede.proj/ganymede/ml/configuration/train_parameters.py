# python
from dataclasses import dataclass
from typing import Tuple
# project
from ganymede.imaging import ImageType


@dataclass
class TrainParameters:
    learning_rate : float

    epochs : int

    train_device_name : str

    input_size : Tuple[int, int]

    input_type : ImageType
    
    batch_size : int
