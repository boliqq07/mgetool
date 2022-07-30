from pathlib import Path

from mgetool.imports import BatchFile
from pymatgen.core import Structure
import numpy as np
import os


bf = BatchFile(r"./")
#
bf.filter_dir_name(include="ed_CONTCAR", layer=-2)
bf.filter_dir_name(include=["Al", "Ca", "Li"], exclude=["Li"], layer=-2)
