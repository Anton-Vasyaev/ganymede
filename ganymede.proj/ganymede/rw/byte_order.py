# python
from enum import IntEnum

from ganymede.augmentation.transformation import augmentation_padding


class ByteOrder(IntEnum):
    LITTLE = 0,
    BIG = 1

    def get_order_str(self) -> str:
        if self == ByteOrder.LITTLE:
            return 'little'
        elif self == ByteOrder.BIG:
            return 'big'
        else:
            raise Exception(f'Unknown code:{int(self)}')
