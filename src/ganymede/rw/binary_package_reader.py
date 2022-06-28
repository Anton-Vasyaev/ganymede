# python
import struct


class BinaryPackageReader:
    def __init__(self, binary_array, byteorder = 'little'):
        self.bin     = binary_array
        self.cur_pos = 0

        self.byteorder = byteorder

    def read_bytes(self, count):
        bin_slice = self.bin[self.cur_pos:self.cur_pos + count]
        self.cur_pos += count

        return bin_slice


    def read_int(self, count, signed=False):
        bin_slice = self.read_bytes(count)

        return int.from_bytes(bin_slice, byteorder=self.byteorder)


    def read_uint8(self): return self.read_int(1, signed=False)


    def read_uint16(self): return self.read_int(2, signed=False)


    def read_uint32(self): return self.read_int(4, signed=False)

    
    def read_uint64(self): return self.read_int(8, signed=False)


    def read_float32(self):
        bin_slice = self.read_bytes(4)

        struct.unpack('f', bin_slice)[0]
