# python
import ctypes
import os
import os.path as p
import platform
from enum   import IntEnum, auto
from ctypes import *

from ganymede.native_libs import NATIVE_LIBS_PATH


def get_bit_design() -> int:
    return ctypes.sizeof(ctypes.c_void_p) * 8


class LibraryModule:
    def __init__(
        self,
        directory_path : str = ''
    ):
        bit_design = get_bit_design()
        if bit_design != 64:
            raise Exception(f'Native libraries not implemented for bit design: x{bit_design}') 

        os_dir = ''
        if os.name == 'nt':
            os_dir = 'win'
        elif os.name == 'posix' and platform.system() == 'Linux':
            os_dir = 'linux'
        else:
            raise NotImplementedError(f'Native libraries not implemented for os:{platform.system()}')

        binary_path = p.join(directory_path, f'{os_dir}{bit_design}')

        if os.name == 'nt':
            import win32api
            win32api.SetDllDirectory(binary_path)
            self.handler = ctypes.CDLL(p.join(binary_path, 'auxml.dll'))




LIBRARY_MODULE = LibraryModule(NATIVE_LIBS_PATH)

__libc = CDLL("msvcrt") if os.name == 'nt' else None

if __libc == None:
    raise Exception('Cannot initialize native C library.')

LIBRARY_C = __libc
