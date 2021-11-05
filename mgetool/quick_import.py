"""
Import dynamic library.
"""

import imp
import importlib.util
import os
import platform
import shutil

from mgetool.tool import def_pwd


def import_module(module_name, path):
    """import module from path"""
    file, path, description = imp.find_module(module_name, [path])
    # Close the .so file after load.
    with file:
        return imp.load_module(module_name, file, path, description)


def import_source(module_name, log_print=False):
    """import module from module.__init__.py path."""
    module_file_path = module_name.__file__
    module_name = module_name.__name__

    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if log_print:
        print(dir(module))
        msg = "The {module_name} module has the following methods:{methods}"
        print(msg.format(module_name=module_name, methods=dir(module)))


def re_build_func(module_name=None, with_html=False):
    """
    Just run build: 'python setup.py build_ext --inplace'.

    Parameters
    ----------
    module_name: str
        just for pyx file.
    with_html:bool
        just for pyx file.

    """
    if os.path.isdir("build"):
        shutil.rmtree("build")
    print("############### build >>> ###############".format(module_name))
    val = os.system("python setup.py build_ext --inplace")
    if with_html:  # just for pyx
        print("build {}.html".format(module_name))
        os.system("cython -a {}.pyx".format(module_name))
    if val != 0:
        raise FileNotFoundError("run 'python setup.py build_ext --inplace' failed due to no 'setup.py' \n"
                                "or error in 'setup.py'. \n"
                                "If without setup.py, please turn to draft to generate 'setup.py' \n")
    else:
        print("########### build end ###############".format(module_name))


def quick_import(module_name, path=None, build=False, suffix=".so", with_html=False, re_build_func=re_build_func,
                 re_build_func_kwargs=None, log_print=False
                 ):
    """
    Import .so file as module.

    Parameters
    ----------
    re_build_func_kwargs:dict
        kwargs for build func
    re_build_func:callable
        build func
    module_name: str
        module_name
    path:
        path
    build:
        build or not.
        default is false and try to find the exist module_name.so or module_name.pyd or module_name.dll.
    suffix:str
        file type [so,pyd] and so on
    with_html:False
        just for cython to check file.

    Returns
    -------
    module

    """
    if suffix is None:
        sys_name = platform.system()
        suffix = ".so" if sys_name == "Linux" else ".pyd"

    def_pwd(path)
    if log_print:
        print("Move to {}".format(os.getcwd()))

    if build:
        if re_build_func_kwargs is None:
            re_build_func(module_name, with_html)
        else:
            re_build_func(**re_build_func_kwargs)

    ext = [i for i in os.listdir() if module_name in i and suffix in i]
    if len(ext) > 0:
        module = import_module(module_name, os.getcwd())
        msg = "The {module_name} module methods:{methods}"
        names = dir(module)
        names = [i for i in names if "__" not in i]
        if log_print:
            print(msg.format(module_name=module_name, methods=names))
        return module
    else:
        raise FileNotFoundError(": There is no './{}.***{}' in '{}',\n".format(module_name, suffix, path),
                                "There are just {},\n".format(os.listdir()),
                                "Please try to build=Ture again.")
