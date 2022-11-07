'''Module of `BinaryPackageReader` '''

# python
import struct
from typing import cast
# project
from ganymede.rw.byte_order import ByteOrder


class BinaryPackageReader:
    ''' Wrapper of bytestring objects for sequential data reading. '''

    binary_array: bytes

    curr_pos: int

    byteorder: ByteOrder

    def __init__(
        self,
        binary_array: bytes,
        byteorder: ByteOrder = ByteOrder.LITTLE
    ):
        '''
        Initialize new object.

        Args:
            binary_array (ByteString): Binary array to be read.
            byteorder (ByteOrder, optional): Byte order of data store. Defaults to ByteOrder.LITTLE.
        '''
        self.binary_array = binary_array
        self.curr_pos = 0

        self.byteorder = byteorder

    def read_bytes(self, count: int) -> bytes:
        '''
        Reads span of byte array and moves the cursor on `count` number of positions.

        Args:
            count (int): Size of span of byte array to be read

        Returns:
            ByteString: Readed span of byte array.
        '''
        bin_slice = self.binary_array[self.curr_pos:self.curr_pos + count]
        self.curr_pos += count

        return bin_slice

    def read_int(self, count: int, signed: bool = False) -> int:
        '''
        Reads integer value value from byte array.

        Args:
            count (int): Size in bytes of readable integer value.
            signed (bool, optional): Flag of signed/unsigned integer format. Defaults to False.

        Returns:
            int: Readed integer value.
        '''
        bin_slice = self.read_bytes(count)

        return int.from_bytes(bin_slice, byteorder=self.byteorder.get_order_str(), signed=signed)

    def read_uint8(self) -> int:
        '''
        Reads 8-bit unsigned integer value.

        Returns:
            int: 8-bit unsigned integer value.
        '''
        return self.read_int(1, signed=False)

    def read_int8(self) -> int:
        '''
        Reads 8-bit signed integer value.

        Returns:
            int: 8-bit signed integer value.
        '''
        return self.read_int(1, signed=True)

    def read_uint16(self) -> int:
        '''
        Reads 16-bit unsigned integer value.

        Returns:
            int: 16-bit unsigned integer value.
        '''
        return self.read_int(2, signed=False)

    def read_int16(self) -> int:
        '''
        Reads 16-bit signed integer value.

        Returns:
            int: 16-bit signed integer value.
        '''
        return self.read_int(2, signed=True)

    def read_uint32(self) -> int:
        '''
        Reads 32-bit unsigned integer value.

        Returns:
            int: 32-bit unsigned integer value.
        '''
        return self.read_int(4, signed=False)

    def read_int32(self) -> int:
        '''
        Reads 32-bit signed integer value.

        Returns:
            int: 32-bit signed integer value.
        '''
        return self.read_int(4, signed=True)

    def read_uint64(self) -> int:
        '''
        Reads 64-bit unsigned integer value.

        Returns:
            int: 64-bit unsigned integer value.
        '''
        return self.read_int(8, signed=False)

    def read_int64(self) -> int:
        '''
        Reads 64-bit signed integer value.

        Returns:
            int: 64-bit signed integer value.
        '''
        return self.read_int(8, signed=True)

    def read_float32(self) -> float:
        '''
        Reads 32-bit floating point value.

        Returns:
            float: 32-bit floating point value.
        '''
        bin_slice = self.read_bytes(4)

        float_val = cast(float, struct.unpack('f', bin_slice)[0])

        return float_val
