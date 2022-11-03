# python 
import struct



class BinaryPackageWriter:
    def __init__(
        self,
        byteorder = 'little' #: Literal['little', 'big'] = 'little'
    ):
        self.bytearray_list = []
        self.byteorder      = byteorder


    def write_bytes(
        self,
        bytes
    ):
        self.bytearray_list.append(bytes)


    def write_int(
        self,
        value     : int,
        length    : int,
        signed    : bool = True, 
    ):
        bytes = value.to_bytes(length, self.byteorder, signed=signed)
        self.write_bytes(bytes)


    def write_uint8(
        self, 
        value     : int
    ): 
        self.write_int(value, 1, False)


    def write_int8(
        self, 
        value     : int
    ): 
        self.write_int(value, 1, True)


    def write_uint16(
        self, 
        value     : int
    ): 
        self.write_int(value, 2, False)


    def write_int16(
        self, 
        value     : int
    ): 
        self.write_int(value, 4, True)


    def write_uint32(
        self, 
        value     : int
    ): 
        self.write_int(value, 4, False)


    def write_int32(
        self, 
        value     : int
    ): 
        self.write_int(value, 4, True)


    def write_uint64(
        self, 
        value     : int
    ): 
        self.write_int(value, 8, False)


    def write_int64(
        self, 
        value     : int
    ): 
        self.write_int(value, 8, True)


    def write_float32(
        self,
        value : float
    ):
        bytes = bytearray(struct.pack('f', value))

        self.bytearray_list.append(bytes)


    def write_float64(
        self,
        value : float
    ):
        bytes = bytearray(struct.pack('f', value))
        self.write_bytes(bytes)


    def form_package(self):
        return bytearray(b'').join(self.bytearray_list)