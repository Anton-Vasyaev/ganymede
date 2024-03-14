# python
from enum import        IntEnum, auto
from typing      import List, Dict, Any
from dataclasses import dataclass, field



class MainDebugTreeGroup:
    PERFORMANCE = auto()
    METRICS     = auto()


@dataclass
class DebugTreeInfo:
    data : Dict[str, Any] = field(default_factory=dict)

    def update(self, data : Dict[str, Any]):
        self.data.update(data)

    def append_value(self, path : str, value : Any):
        pathes = path.split('.')

        way_data = self.data

        for idx in range(len(pathes)):
            curr_path = pathes[idx]

            if idx == len(pathes) - 1:
                if not curr_path in way_data:
                    way_data[curr_path] = value
                else:
                    raise Exception(f'data on path:{path} is already exist.')
            else:
                if not curr_path in way_data:
                    way_data[curr_path] = dict()

                way_data = way_data[curr_path]

    def append_tree(self, path : str, tree : Any):
        self.append_value(path, tree.data)



