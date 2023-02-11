# -*- coding: utf-8 -*-
import os
from typing import Union, List

import path
import pathlib


# @Time  : 2023/2/11 15:24
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

def find_leaf_path(root_pt: Union[str, path.Path, os.PathLike, pathlib.Path], abspath: bool = False) -> List[path.Path]:
    """
    Find the leaf path.

    Args:
        abspath: bool, return abspath or not.
        root_pt: pt: (str, path.Path, os.PathLike,pathlib.Path), path.

    Returns:
        paths: (list), list of sub leaf path.

    """
    if not isinstance(root_pt, path.Path):
        root_pt = path.Path(root_pt)

    sub_disk = list(root_pt.walkdirs())

    if abspath:
        sub_disk = [i.abspath() for i in sub_disk]

    par_disk = [i.parent for i in sub_disk]

    par_disk = set(par_disk)

    res = set(sub_disk) - par_disk

    res = list(res)
    res.sort()

    return res

if __name__ == '__main__':
    res = find_leaf_path("../..", abspath=True)
