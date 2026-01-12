# python
from typing import List
# 3rd party
from pathlib import Path
# project
import ganymede.core as g_core


def search_files(
    root               : str, 
    pattern            : str = '*', 
    alpha_numeric_sort : bool = True,
    recursive          : bool = False,
) -> List[str]:
    path = Path(root)

    search_method = path.rglob if recursive else path.glob

    files : List[str] = []
    for p in search_method(pattern):
        if p.is_file():
            files.append(str(p))

    if alpha_numeric_sort:
        g_core.alpha_numeric_sort(files)

    return files


def iterate_dir(
    root               : str,
    pattern            : str = '*',
    alpha_numeric_sort : bool = True
) -> List[str]:
    path = Path(root)

    children : List[str] = []
    for p in path.glob(pattern):
        children.append(str(p))

    if alpha_numeric_sort:
        g_core.alpha_numeric_sort(children)

    return children