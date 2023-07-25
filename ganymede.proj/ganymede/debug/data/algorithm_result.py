# python
from dataclasses import dataclass, field
from typing      import Optional, Generic, TypeVar
# project
from .debug_info import DebugInfo


ResultT = TypeVar('ResultT')


@dataclass
class AlgorithmData(Generic[ResultT]):
    result : ResultT

    debug_info : DebugInfo = field(default_factory=DebugInfo)


    def update_debug_info(self, debug_info : DebugInfo):
        self.debug_info.update(debug_info)

