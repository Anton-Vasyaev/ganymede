# python
import re
# 3rd party
# 3rd party
from pathlib import Path


def rsearch_files(root, rglob_rule = '*', recursive = False):
    if not type(root) is list: root = [root] 

    video_files = []
    for root in root:
        path = Path(root)
        for p in path.rglob(rglob_rule):
            if p.is_file():
                video_files.append(str(p))

    return video_files