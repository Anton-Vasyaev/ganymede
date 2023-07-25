# python
from dataclasses import dataclass
from typing      import Tuple

@dataclass
class ImageShape:
    width : int

    height : int

    channels : int


    def decompose(self) -> Tuple[int, int, int]:
        return self.width, self.height, self.channels