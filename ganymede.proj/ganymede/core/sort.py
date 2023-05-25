# python
import re
from typing import List, Any, Union


def tryint(s : str) -> Union[str, int]:
    try:
        return int(s)
    except:
        return s


def alphanum_key(s : str) -> List[Union[str, int]]:
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def alpha_numeric_sort(
    l       : List[str],
    reverse : bool = False
) -> List[str]:
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key, reverse=reverse)
