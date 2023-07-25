# python
from dataclasses import dataclass, field
from typing      import Any, cast
# project
from .debug_image_info import DebugImageInfo


@dataclass
class DebugInfo:
    image : DebugImageInfo = field(default_factory=DebugImageInfo)

    def update(self, debug_info_a : Any):
        debug_info = cast(DebugInfo, debug_info_a)
        self.image.update(debug_info.image)