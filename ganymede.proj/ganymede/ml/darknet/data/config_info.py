# python
from dataclasses import dataclass, field
from typing      import List, Dict, Any, Optional, cast


@dataclass
class ConfigParameter:
    name : str

    value : str

    line_number : int


@dataclass
class ConfigBlock:
    name : str

    params : Dict[str, ConfigParameter]

    line_number : int

    parent_info : Optional[Any] = field(default=None)


    def get_file_path(self) -> Optional[str]:
        if self.parent_info is None:
            return None
        else:
            parent_info_t = cast(ConfigInfo, self.parent_info)
            return parent_info_t.path


@dataclass
class ConfigInfo:
    blocks : List[ConfigBlock] 

    path : Optional[str]


    def __init__(self, blocks : List[ConfigBlock], path : Optional[str]):
        self.blocks = blocks
        self.path   = path

        for block in self.blocks:
            block.parent_info = self