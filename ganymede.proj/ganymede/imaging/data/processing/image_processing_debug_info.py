# python
from dataclasses import dataclass, field
from typing      import Dict
# 3rd party
import numpy as np
# project
from .image_processing_draw_info import ImageProcessingDrawInfo


@dataclass
class ImageProcessingDebugInfo:
    debug_frames : Dict[str, np.ndarray] = field(default_factory=dict)

    debug_draws : Dict[str, ImageProcessingDrawInfo] = field(default_factory=dict)