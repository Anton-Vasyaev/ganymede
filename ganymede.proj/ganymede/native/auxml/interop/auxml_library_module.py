# python
import ctypes
import os
import os.path as p
import platform
from enum   import IntEnum, auto
from ctypes import *

from ganymede.native import NATIVE_LIBS_PATH


def get_bit_design() -> int:
    return ctypes.sizeof(ctypes.c_void_p) * 8


POSIX_LOAD_LIBRARIES = []

class LibraryModule:
    handler : CDLL

    def __init__(
        self,
        directory_path : str = ''
    ):
        try:
            bit_design = get_bit_design()
            if bit_design != 64:
                print(f'Native libraries not implemented for bit design: x{bit_design}')
                return

            os_dir = ''
            if os.name == 'nt':
                os_dir = 'win'
            elif os.name == 'posix' and platform.system() == 'Linux':
                os_dir = 'linux'
            else:
                print(f'Not load native libraries for os:{platform.system()}')
                return

            binary_path = p.join(directory_path, f'{os_dir}{bit_design}')

            if os.name == 'nt':
                import win32api
                win32api.SetDllDirectory(binary_path)
                self.handler = ctypes.CDLL(p.join(binary_path, 'auxml.dll'))
            elif os.name == 'posix' and platform.system() == 'Linux':
                POSIX_LOAD_LIBRARIES.append(CDLL('libstdc++.so.6'))
                self.handler = CDLL(p.join(binary_path, 'libauxml.so'))
        except Exception as exc:
            print(f'Failed to load native libraries:{exc}')
            

LIBRARY_MODULE = LibraryModule(NATIVE_LIBS_PATH)

__libc = CDLL('msvcrt') if os.name == 'nt' else CDLL('libc.so.6')

if __libc is None:
    raise Exception('Cannot initialize native C library.')

LIBRARY_C = __libc
