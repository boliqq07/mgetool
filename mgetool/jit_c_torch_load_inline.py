"""This is one reserved script, Copy it to your module for use."""
import imp
import os
import platform
from pathlib import Path

import torch
import torch.utils.cpp_extension


def import_module_from_library(module_name, path, is_python_module):
    file, path, description = imp.find_module(module_name, [path])
    # Close the .so file after load.
    with file:
        if is_python_module:
            return imp.load_module(module_name, file, path, description)
        else:
            torch.ops.load_library(path)


source = """
#include <torch/extension.h>
#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}
"""


def inline_c_torch(source_cpp=source, module_name="segment_method", temps="temps",
                   suffix=None, functions=['d_sigmoid', ]):
    """
    torch.utils.cpp_extension.load_inline, just jump build if exist.

    Parameters
    ----------
    source_cpp:str
        all the cpp source text but not file name.
    temps: str
        Add one temps dir and copy all the file to this disk to escape pollution.
    module_name:str
        name of module
    suffix:str
        module file type.
    functions:str,tuple
        name of function in cpp.
    """
    name = module_name

    if platform.system() == "Windows":
        ext = "dll"
    else:
        ext = "so"

    if suffix:
        ext = suffix

    name_dir = temps + name
    MODULE_DIR = Path(__file__).parent.absolute()
    MODULE_DIR_NAME_DIR = MODULE_DIR / name_dir

    if os.path.isdir(MODULE_DIR_NAME_DIR) and os.path.isfile(MODULE_DIR_NAME_DIR / "{}.{}".format(name, ext)):
        mod = import_module_from_library(name, MODULE_DIR_NAME_DIR, True)

    else:
        if not os.path.isdir(MODULE_DIR_NAME_DIR):
            os.mkdir(MODULE_DIR_NAME_DIR)
        mod = torch.utils.cpp_extension.load_inline(
            name=name,
            cpp_sources=source_cpp,
            verbose=True,
            build_directory=MODULE_DIR_NAME_DIR,
            is_python_module=True,
            functions=functions
        )
    return mod


if __name__ == '__main__':
    inputs = torch.tensor([1.2, 3, 4, 5, 6, 0.7, 9], requires_grad=True, device="cpu")
    node_bond_idx = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    # b = mod.merge_idx(inputs, node_bond_idx, "mean")
