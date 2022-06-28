# python
from dataclasses import dataclass
from typing import Any
# 3rd party
import numpy as np


@dataclass
class AugmentationData:
    image  : np.ndarray
    points : Any