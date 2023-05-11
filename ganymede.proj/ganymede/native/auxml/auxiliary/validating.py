import ctypes
from ctypes import *

from ganymede.native.auxml.interop.data import return_status
from ganymede.native.auxml.interop.api  import api_done_return_status
from ganymede.native.auxml.interop import LIBRARY_C

def validate_return_status(
    status : return_status
):
    err_msg_s = ''

    if status.correct_status != 1:
        msg_len = LIBRARY_C.strlen(status.error_message)
        char_arr = (c_char * (msg_len + 1))()
        char_arr[-1] = 0
        memmove(byref(char_arr), status.error_message, msg_len)

        err_msg_s = bytearray(char_arr).decode('ascii')

    api_done_return_status(status)
    
    if err_msg_s != '':
        raise Exception(f'Native exception in auxml:{err_msg_s}.')