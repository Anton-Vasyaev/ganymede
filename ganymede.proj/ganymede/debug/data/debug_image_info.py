# python
from dataclasses import dataclass, field
from typing      import Dict, Any, cast
# 3rd party
import numpy as np
# project
from .debug_draw_info import DebugDrawInfo


@dataclass
class DebugImageInfo:
    images : Dict[str, np.ndarray] = field(default_factory=dict)

    draws : Dict[str, DebugDrawInfo] = field(default_factory=dict)


    def update(self, image_info_a : Any):
        images_info = cast(DebugImageInfo, image_info_a)

        self.images.update(images_info.images)
        self.draws.update(images_info.draws)
