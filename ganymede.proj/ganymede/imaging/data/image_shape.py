# python
from dataclasses import dataclass
from typing      import Tuple

@dataclass
class ImageShape:
    width : int

    height : int

    channels : int


    def decompose(self) -> Tuple[int, int, int]:
        '''
        Returns int tuple of image shape in order: width, height, channels

        Returns:
            Tuple[int, int, int]: int tuple of image shape
        '''
        
        return self.width, self.height, self.channels