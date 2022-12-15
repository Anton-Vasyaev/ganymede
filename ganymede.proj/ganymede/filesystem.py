# python
import re
from typing import List, Union
# 3rd party
from pathlib import Path
# project
import ganymede.core as g_core


def rsearch_files(root : Union[str, List[str]], rglob_rule = '*', recursive = False):
    if isinstance(root, str):
        root_list = [root]
    else:
        root_list = root

    files = []
    for root in root_list:
        path = Path(root)
        for p in path.rglob(rglob_rule):
            if p.is_file():
                files.append(str(p))

    g_core.alpha_numeric_sort(files)

    return files