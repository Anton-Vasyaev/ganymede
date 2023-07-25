# python
from dataclasses import dataclass
from typing      import Tuple


@dataclass
class ImageSize:
    width : int

    height : int

    def decompose(self) -> Tuple[int, int]:
        return self.width, self.height