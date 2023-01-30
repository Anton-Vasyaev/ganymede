# python
from dataclasses import dataclass
from typing import cast


@dataclass
class DarknetForwardSize:
    width : int

    height : int

    channels : int


    def __eq__(self, other) -> bool:
        o = cast(DarknetForwardSize, other)

        return self.width == o.width and self.height == o.height and o.channels == o.channels


    def __str__(self) -> str:
        return f'({self.channels}, {self.height}, {self.width})'


    def detail_str(self) -> str:
        return f'{self.__str__()} (channels, height, width).'