# python
import re
from typing import List
# 3rd party
from pathlib import Path
# project
import ganymede.core as g_core


def rsearch_files(root : str, rglob_rule : str = '*', recursive = False) -> List[str]:
    path = Path(root)

    search_method = path.glob if recursive else path.glob

    files : List[str] = list()
    for p in search_method(rglob_rule):
        if p.is_file():
            files.append(str(p))

    g_core.alpha_numeric_sort(files)

    return files