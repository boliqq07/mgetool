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


def inline_c_torch(module_name="segment_method", source_cpp="inst.cpp", suffix=None, temps="temps",
                   functions=["funcs",]):
    """
    torch.utils.cpp_extension.load, just jump build if exist.

    Parameters
    ----------
    source_cpp:str
        cpp file name.
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
    MODULE_DIR = Path(source_cpp).parent.absolute()
    MODULE_DIR_NAME_DIR = MODULE_DIR / name_dir

    if os.path.isdir(MODULE_DIR_NAME_DIR) and os.path.isfile(MODULE_DIR_NAME_DIR / "{}.{}".format(name, ext)):
        mod = import_module_from_library(name, MODULE_DIR_NAME_DIR, True)

    else:
        if not os.path.isdir(MODULE_DIR_NAME_DIR):
            os.mkdir(MODULE_DIR_NAME_DIR)

        with open(source_cpp, "a+") as f:
            strs = f.readlines()
            strs = strs[-20:] if len(strs) > 20 else strs
            if any([True if 'PYBIND11_MODULE' in i else False for i in strs]):
                pass
            else:
                if functions is not None:
                    module_def = [" ", ]
                    module_def.append('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {')
                    if isinstance(functions, str):
                        functions = [functions]
                    if isinstance(functions, list):
                        # Make the function docstring the same as the function name.
                        functions = dict((f, f) for f in functions)
                    elif not isinstance(functions, dict):
                        raise ValueError(
                            "Expected 'functions' to be a list or dict, but was {}".format(
                                type(functions)))
                    for function_name, docstring in functions.items():
                        module_def.append('m.def("{0}", {0}, "{1}");'.format(function_name, docstring))
                    module_def.append('}')

                    f.writelines(module_def)

        mod = torch.utils.cpp_extension.load(
            name=name,
            sources=[source_cpp, ],
            verbose=True,
            build_directory=MODULE_DIR_NAME_DIR,
            is_python_module=True,
        )

    return mod


if __name__ == '__main__':
    a = inline_c_torch(module_name="segment_method", source_cpp="../inst/speedup_torch.cpp",
                       functions=["d_sigmoid"])

    # inputs = torch.tensor([1.2, 3, 4, 5, 6, 0.7, 9], requires_grad=True, device="cpu")
    # node_bond_idx = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    # b = mod.merge_idx(inputs, node_bond_idx, "mean")
