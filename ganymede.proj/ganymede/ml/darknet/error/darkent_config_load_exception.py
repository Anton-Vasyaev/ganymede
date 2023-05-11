# python
from typing import Optional, List

class DarknetConfigLoadException(Exception):
    reason : str

    cfg_path : Optional[str]

    line : Optional[int]

    def __init__(
        self,
        reason   : str,
        cfg_path : Optional[str] = None,
        line     : Optional[int] = None
    ):
        self.reason   = reason
        self.cfg_path = cfg_path
        self.line     = line


    def __str__(self):
        message_list : List[str] = list()

        message_list.append(f'Error in darknet configuration.')

        if not self.cfg_path is None:
            message_list.append(f' File \'{self.cfg_path}\'.')

        if not self.line is None:
            message_list.append(f' Line :{self.line}')

        message_list.append('.')

        message_list.append(f' Reason:{self.reason}.')

        return ''.join(message_list)