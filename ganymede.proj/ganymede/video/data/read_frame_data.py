# python
from dataclasses import dataclass
from datetime    import datetime
# 3rd party
import numpy as np


@dataclass
class ReadFrameData:
    frame : np.ndarray

    timestamp : int