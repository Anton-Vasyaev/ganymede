# python
from dataclasses import dataclass, field
from typing      import Optional, Generic, TypeVar
# project
from .image_processing_debug_info import ImageProcessingDebugInfo


ResultT = TypeVar('ResultT')


@dataclass
class ImageProcessingResult(Generic[ResultT]):
    result : ResultT

    debug_info : Optional[ImageProcessingDebugInfo] = field(default=None)


    def update_debug_info(self, debug_info : Optional[ImageProcessingDebugInfo]):
        if not debug_info is None:
            if self.debug_info is None:
                self.debug_info = ImageProcessingDebugInfo()

            debug_info.debug_frames.update(debug_info.debug_frames)
            
            debug_info.debug_draws.update(debug_info.debug_draws)