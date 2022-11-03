# python
import re
# 3rd party
from pathlib import Path
# project
import ganymede.core as g_core


def rsearch_files(root, rglob_rule = '*', recursive = False):
    if not type(root) is list: root = [root] 

    files = []
    for root in root:
        path = Path(root)
        for p in path.rglob(rglob_rule):
            if p.is_file():
                files.append(str(p))

    g_core.alpha_numeric_sort(files)

    return files