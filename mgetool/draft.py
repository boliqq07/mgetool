"""
Quick build for pyx and cpp file.
used for Rapid debugging.

After build=True, the temporary files such as setup.py in local, could be re-write further.
and select the required parts text  and add to your module main ``setup.py`` file.

Include:
---------
``DraftPybind11``
    For cpp (optional, with ``Eigen``).
``DraftPyx``
    For pyx for cython.
``DraftTorch``
    For cpp (optional, with ``torch``)
``TorchJit``
    For cpp (with ``torch``)
``TorchJitInLine``
    For cpp text (with ``torch``)

Examples
---------
>>> from mgetool.draft import DraftPyx

>>> bd = DraftPyx("hello.pyx")
>>> bd.write()
>>> a= bd.quick_import(build=True, with_html=True)
>>> # bd.remove()

Examples
---------

>>> from mgetool.draft import DraftPybind11

>>> bd = DraftPybind11("speedup_pybind11.cpp",pybind11_path=r"/home/iap13/wcx/pybind11",
>>>                    eigen_path=r"/home/iap13/wcx/eigen-3.3.9")
>>> bd.write(functions=["cubic","fcc","bcc"])
>>> a= bd.quick_import(build=True)
>>> # bd.remove()


Examples
---------

>>> from mgetool.draft import TorchJit
>>> bd = TorchJit("speedup_torch.cpp",warm_start=True)
>>> a= bd.quick_import(build=False)

"""

import os
import shutil
import warnings
from abc import abstractmethod
from pathlib import Path

from mgetool.quick_import import quick_import
from mgetool.tool import def_pwd, get_name_without_suffix


class BaseDraft:
    """Base draft for quick test."""

    def __init__(self, file, path=None, temps="temps", warm_start=False, only_file=True, log_print=True):
        """
        Add one temps dir and copy all the file to this disk to escape pollution.

        Parameters
        ----------
        file:str
            file name without path.
        path:str
            path of file.
        temps: str
            Add one temps dir and copy all the file to this disk to escape pollution.
        warm_start:bool
            start from exist file.
        only_file:bool
            just copy the source file to temps.
        log_print:bool
            print the log or not.
        """
        self.log_print = log_print
        self.init_path = os.getcwd()
        # check file
        if path:
            assert r"/" not in file, "Path must in one of `path` parameter or the `filename`."
            assert r"\\" not in file, "Path must in one of `path` parameter or the `filename`."
        else:
            if "/" in file or "\\" in file:
                path = Path(file).parent
                file = os.path.split(file)[-1]

        self.file = file
        self.check_suffix(self.file)

        if path is not None:
            def_pwd(path)
        if os.path.isfile(file):
            pass
        else:
            raise IOError("No file named {} in {}, please re-site your path".format(file, os.getcwd()))

        MODULE_DIR = Path().absolute()
        # temps:
        if temps:
            if warm_start:
                if not os.path.isdir(temps):
                    raise FileNotFoundError("Try to warm start but without no exist {} found".format(temps))
            else:

                files = os.listdir()
                if temps in files:
                    files.remove(temps)
                if only_file:
                    files = [self.file, ]

                if not os.path.isdir(temps):
                    os.mkdir(temps)
                else:
                    warnings.warn("There is exist {temps}. Duplicate files will be overwritten."
                                  "please use remove() to delete temporary file after test.".format(temps=temps))
                for i in files:
                    if os.path.isdir(i):
                        if os.path.isdir(MODULE_DIR / temps / i):
                            shutil.rmtree(MODULE_DIR / temps / i)
                        shutil.copytree(i, MODULE_DIR / temps / i)
                    else:
                        shutil.copy(i, temps)

            self.temps = temps
            self.path = MODULE_DIR / temps

        else:
            self.temps = None
            self.path = MODULE_DIR
        # check module_name
        module_name = get_name_without_suffix(os.path.split(self.file)[-1])
        self.module_name = module_name
        self.build = True
        os.chdir(self.init_path)

    @abstractmethod
    def _suffix(self):
        return "pyx"

    def check_suffix(self, file):
        file = file.split(".")[-1]
        if file == self._suffix():
            pass
        else:
            raise NameError("Just accept {} file".format(self._suffix()))

    def write(self, *args, **kwargs):
        """
        Write setup.py (and optional add tail or head to cpp file.)

        args and kwargs in text_tail:

        Parameters
        ----------
        args:
            parameter for tail.
        kwargs:
            parameter for tail.

            doc="module Document". Document of module.

            functions=["fcc", "bcc", "cubic"]. Method of module, keep same with cpp functions.

        """
        os.chdir(self.path)
        # setup.py
        try:
            setup = self.text_setup(self.module_name, self.file)
            if setup is not None:
                print("Write 'setup.py' for {}.".format(self.module_name))
                setup = self.text_setup(self.module_name, self.file)
                with open(r"./setup.py", "w") as f:
                    f.write(setup)
                print("Make new 'setup.py' successfully.")
            else:
                raise NotImplementedError
        except NotImplementedError:
            pass

        try:
            head = self._text_head()
            if head is not None:
                print("Add head text to {}".format(self.file))

                if self._suffix() == "cpp" or self._suffix() == "c":

                    with open(self.file, "r+") as f:
                        lines = f.readlines()
                        i = 0
                        z = 0
                        while "#" == lines[i][0] or z == 0:
                            if "#" != lines[i][0]:
                                z = 0
                            else:
                                z = 1
                            i += 1
                        head.reverse()
                        [lines.insert(i, h) for h in head if h not in lines]
                        f.seek(0)
                        f.truncate()
                        f.writelines(lines)

                elif self._suffix() == "pyx":
                    with open(self.file, "r+") as f:
                        lines = f.readlines()
                        if "language_level" in lines[0]:
                            pass
                        else:
                            [lines.insert(0, h) for h in head if h not in lines]
                            f.seek(0)
                            f.truncate()
                            f.writelines(lines)
                else:
                    raise NotImplementedError

        except NotImplementedError:
            pass

        try:
            tail = self._text_tail(self.module_name, *args, **kwargs)
            if tail is not None:
                print("Add tail text to {}".format(self.file))
                with open(self.file, "a+") as f:
                    f.seek(0)
                    strs = f.readlines()
                    strs30 = strs[-30:] if len(strs) > 30 else strs
                    if any([True if 'PYBIND11_MODULE' in i else False for i in strs30]):
                        strs_new = []
                        for i in strs:
                            if 'PYBIND11_MODULE' not in i:
                                strs_new.append(i)
                            else:
                                break
                        f.seek(0)
                        f.truncate()
                        f.writelines(strs_new)

                    f.write(tail)
        except NotImplementedError:
            pass

        os.chdir(self.init_path)

        # end

    @abstractmethod
    def _text_head(self, *args, **kwargs):
        """
        Write head text in cpp file automatically for temporary test.
        for pybind11.
        """

    @abstractmethod
    def _text_tail(self, *args, **kwargs):
        """
        Write tail text in cpp file automatically for temporary test.
        for pybind11.
        """

    @abstractmethod
    def _text_setup(self, *args, **kwargs):
        """"""

    def text_setup(self, *args, **kwargs):
        """
        Write setup.py file for f_pyx automatically for temporary test.
        And the setup.py could be costumed further by yourself.
        All the data is in the same folder.
        """
        return self._text_setup(*args, **kwargs)

    def quick_import(self, build=False, suffix=".so", with_html=False):
        os.chdir(self.path)
        self.build = build
        if self._suffix() != "pyx":
            with_html = False
        mod = quick_import(self.module_name, path=self.path, build=build, suffix=suffix, with_html=with_html,
                           log_print=self.log_print)
        os.chdir(self.init_path)
        if self.log_print:
            print("Move back to {}".format(self.init_path))
        return mod

    def remove(self, numbers=None):
        os.chdir(self.path)
        """remove files, beyond retrieve, carefully use!!!"""
        if self.temps is None and numbers is None:
            warnings.warn("Difficult to determine which file to delete. please pass the numbers, such as [0,1,2],"
                          "Please confirm the serial number")
            print("Exist file: {}")
            for i in zip(range(len(os.listdir())), os.listdir()):
                print(i)
        elif self.temps is None and numbers is not None:
            files = [os.listdir()[i] for i in numbers]
            for i in files:
                if os.path.isdir(i):
                    print("dir {} is deleted.".format(i))
                    shutil.rmtree(i)
                else:
                    print("file {} is deleted.".format(i))
                    os.remove(i)
            if self.build is False:
                print(
                    "! Please commented out or delete this line in file to prevent mistaken delete in next time running\n"
                    "Such as:\n"
                    " >>> # bd.remove([6,7,8,9,10])")
            else:
                print("Delete end")
        else:
            shutil.rmtree(self.path)
            print("Delete end")

        os.chdir(self.init_path)


class DraftPybind11(BaseDraft):
    """
    Build for cpp by Pybind11.

    Examples
    -----------

    >>> from mgetool.draft import DraftPybind11

    >>> bd = DraftPybind11("speedup_pybind11.cpp", pybind11_path=r"/home/iap13/wcx/pybind11",
    >>>                    eigen_path=r"/home/iap13/wcx/eigen-3.3.9")
    >>> bd.write(functions=["cubic","fcc","bcc"])
    >>> a= bd.quick_import(build=True)
    >>> # bd.remove()
    """

    def __init__(self, *args, pybind11_path=r"/home/iap13/wcx/pybind11",
                 eigen_path=r"/home/iap13/wcx/eigen-3.3.9", **kwargs):
        """

        Parameters
        ----------
        args
        pybind11_path:
            path of pybind11
        eigen_path:
            path of eigen_path, if you are not using it, please pass '.'
        kwargs
        """
        super(DraftPybind11, self).__init__(*args, **kwargs)

        self.pybind11_path = pybind11_path

        self.eigen_path = "." if eigen_path is None else eigen_path
        if self.log_print:
            print("Check and re-set you path:")
            print("pybind11_path:{}".format(pybind11_path))
            print("eigen_path:{}".format(pybind11_path))

    def _suffix(self):
        return "cpp"

    def _text_setup(self, module_name, file):
        pybind11_path = self.pybind11_path
        eigen_path = self.eigen_path
        setup_text = """
import warnings
warnings.filterwarnings(
    "ignore", "Distutils was imported before Setuptools. This usage is discouraged and may exhibit undesirable behaviors or errors. Please use Setuptools' objects directly or at least import Setuptools first.",  UserWarning, "setuptools.distutils_patch")

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super(BuildExt, self).build_extensions()
        
        
if "cpp" in "{f_cpp}":
    b_ext = BuildExt
else:
    b_ext = build_ext


ext_modules = []

# pybind11
include_dirs = [r'{pybind11_path}',] 
if "{eigen_path}" == "None" or "{eigen_path}" == ".":
    pass
else:
    include_dirs.append(r'{eigen_path}')  # if need

ext_modules.append(Pybind11Extension("{module_name}", ["{f_cpp}"], language="c++", include_dirs=include_dirs))


def main():
    setup(name="{module_name}",
          version="0.0.1",
          cmdclass={{"build_ext": b_ext}},
          description="Python / C library ",
          author="boliqq07",
          author_email="98988989@qq.com",
          packages=find_packages(exclude=[".tests", ".tests.", "tests.", "tests"]),
          platforms=[
              "Windows",
              "Unix",
          ],
          classifiers=[
              "Development Status :: 4 - Beta",
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
              "Natural Language :: English",
              "Operating System :: Microsoft :: Windows",
              "Operating System :: Unix",
              "Programming Language :: Python :: 3.6",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.8",
              "Programming Language :: Python :: 3.9",
          ],
          include_dirs=None,
          # include_dirs=include_dirs,
          ext_modules=ext_modules)

if __name__ == "__main__":
    main()
# `Next in cmd`
# local:
# python setup.py build_ext --inplace
# total:
# python setup.py bdist_wheel

        """.format(module_name=module_name, f_cpp=file, pybind11_path=pybind11_path, eigen_path=eigen_path)

        return setup_text

    def _text_head(self):
        return ["#include <pybind11/pybind11.h>\n",
                "#include <pybind11/stl.h>\n",
                "namespace py = pybind11;\n"
                ]

    def _text_tail(self, module_name, doc="module Document",
                   functions=("fcc", "bcc", "cubic")) -> str:
        """
        Add following code to f_cpp file, Just for function, rather than class.

        Add following code to f_cpp file.''

        PYBIND11_MODULE(module_name, m) {
        m.doc() = "module Document"; // optional module docstring
        m.def("cubic", &cubic, "Document");
        m.def("fcc", &fcc, "Document");
        m.def("bcc", &bcc, "Document");
        }
        ''


        Parameters
        ----------
        module_name:str
            module name, please make sure the name just contain [a-z][0-9] and _. such as "a", default using the name of f_cpp.
        doc:str
            module document.
        functions:list of str
            function name, the names must be function in cpp file.
        """

        func_str = ['m.def("{i}", &{i}, "function Document"); // functions'.format(i=i) for i in functions]
        func_str = "\n".join(func_str)

        PYBIND11_MODULE = r"""
PYBIND11_MODULE({module_name}, m) {{
m.doc() = "{doc}"; // optional module docstring
{func_str}
}}""".format(module_name=module_name, doc=doc, func_str=func_str)
        return PYBIND11_MODULE


class DraftPyx(BaseDraft):
    """
    Build for Pyx.

    Examples
    ---------
    >>> from mgetool.draft import DraftPyx

    >>> bd = DraftPyx("hello.pyx")
    >>> bd.write()
    >>> a= bd.quick_import(build=True, with_html=True)
    >>> # bd.remove()

    Examples
    ---------
    >>> from mgetool.draft import DraftPyx

    >>> bd = DraftPyx("hello.pyx",language="c++")
    >>> bd.write()
    >>> a= bd.quick_import(build=True, with_html=True)
    >>> # bd.remove()
    """

    def __init__(self, *args, language="c", **kwargs):
        super(DraftPyx, self).__init__(*args, **kwargs)
        self.language = language

    def _text_setup(self, module_name, file):
        setup_text = """
import warnings
warnings.filterwarnings(
    "ignore", "Distutils was imported before Setuptools. This usage is discouraged and may exhibit undesirable behaviors or errors. Please use Setuptools' objects directly or at least import Setuptools first.",  UserWarning, "setuptools.distutils_patch")
from setuptools import setup
from Cython.Build import cythonize

import numpy

include_path = [numpy.get_include()]  # just for cimport numpy

def main():

    setup(
        name='{module_name}',
        ext_modules=cythonize('{f_pyx}',include_path=include_path,language='{language}'),)

if __name__ == "__main__":
    main()
# `Next in cmd`
# local:
# python setup.py build_ext --inplace
# total:
# python setup.py bdist_wheel

""".format(module_name=module_name, f_pyx=file, language=self.language)

        return setup_text

    def _text_head(self, *args, **kwargs):
        return [
        ]


class DraftTorch(BaseDraft):
    """
    Torch draft.

    Examples:
    -----------
    >>> from mgetool.draft import DraftTorch
    >>>
    >>> bd = DraftTorch("speedup_torch.cpp")
    >>> bd.write(functions=["d_sigmoid",])
    >>> a= bd.quick_import(build=True)
    >>> bd.remove()

    """

    def __init__(self, *args, pybind11_path=r'/home/iap13/wcx/pybind11', **kwargs):
        super(DraftTorch, self).__init__(*args, **kwargs)
        self.pybind11_path = pybind11_path
        if self.log_print:
            print("Check and re-set you path:")
            print("pybind11_path:{}".format(pybind11_path))

    def _suffix(self):
        return "cpp"

    def _text_tail(self, module_name, doc="module Document",
                   functions=("fcc", "bcc", "cubic")) -> str:
        """
        Add following code to f_cpp file, Just for function, rather than class.

        Add following code to f_cpp file.''

        PYBIND11_MODULE(module_name, m) {
        m.doc() = "module Document"; // optional module docstring
        m.def("cubic", &cubic, "Document");
        m.def("fcc", &fcc, "Document");
        m.def("bcc", &bcc, "Document");
        }
        ''


        Parameters
        ----------
        module_name:str
            module name, please make sure the name just contain [a-z][0-9] and _. such as "a", default using the name of f_cpp.
        doc:str
            module document.
        functions:list of str
            function name, the names must be function in cpp file.
        """

        func_str = ['m.def("{i}", &{i}, "function Document"); // functions'.format(i=i) for i in functions]
        func_str = "\n".join(func_str)

        PYBIND11_MODULE = r"""
PYBIND11_MODULE({module_name}, m) {{
m.doc() = "{doc}"; // optional module docstring
{func_str}
}}""".format(module_name=module_name, doc=doc, func_str=func_str)
        return PYBIND11_MODULE

    def _text_head(self):
        return ["#include <pybind11/pybind11.h>\n",
                "#include <pybind11/stl.h>\n",
                "namespace py = pybind11;\n"
                ]

    def text_setup(self, module_name, f_cpp):
        pybind11_path = self.pybind11_path

        setup_text = """
import warnings
warnings.filterwarnings("ignore", "Distutils was imported before Setuptools. This usage is discouraged and may exhibit undesirable behaviors or errors. Please use Setuptools' objects directly or at least import Setuptools first.",  UserWarning, "setuptools.distutils_patch")

from setuptools import setup, find_packages
from torch.utils import cpp_extension


class BuildExt(cpp_extension.BuildExtension):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super(BuildExt, self).build_extensions()
        
        
if "cpp" in "{f_cpp}":
    b_ext = BuildExt
else:
    b_ext = cpp_extension.BuildExtension

ext_modules = []

# pybind11
include_dirs = cpp_extension.include_paths()
include_dirs.append(r'{pybind11_path}')

library_paths = cpp_extension.library_paths()
extra_link_args = []
for i in library_paths:
    extra_link_args.append('-Wl,-rpath={{}}'.format(i))

ext_modules.append(cpp_extension.CppExtension("{module_name}", ["{f_cpp}"], language="c++",
include_dirs=include_dirs,extra_link_args=extra_link_args))

def main():
    setup(name="{module_name}",
          version="0.0.1",
          cmdclass={{"build_ext": b_ext}},
          description="Python / C library ",
          author="boliqq07",
          author_email="98988989@qq.com",
          packages=find_packages(exclude=[".tests", ".tests.", "tests.", "tests"]),
          platforms=[
              "Windows",
              "Unix",
          ],
          classifiers=[
              "Development Status :: 4 - Beta",
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
              "Natural Language :: English",
              "Operating System :: Microsoft :: Windows",
              "Operating System :: Unix",
              "Programming Language :: Python :: 3.6",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.8",
              "Programming Language :: Python :: 3.9",
          ],
          include_dirs=None,
          # include_dirs=include_dirs,
          ext_modules=ext_modules)

if __name__ == "__main__":
    main()
# `Next in cmd`
# local:
# python setup.py build_ext --inplace
# total:
# python setup.py bdist_wheel

            """.format(module_name=module_name, f_cpp=f_cpp, pybind11_path=pybind11_path)

        return setup_text


class TorchJit(BaseDraft):
    """This type is one JIT of torch, with no setup.py file.

    Examples
    -----------
    >>> from mgetool.draft import TorchJit

    >>> bd = TorchJit("speedup_torch.cpp")
    >>> bd.write(functions=["d_sigmoid",])
    >>> a= bd.quick_import(build=True)
    >>> bd.remove()
    """

    def __init__(self, file, path=None, temps="temps", warm_start=False, **kwargs):
        super(TorchJit, self).__init__(file, path=path, temps=temps, warm_start=warm_start, **kwargs)
        try:
            import ninja, torch
        except ImportError:
            raise ImportError("ninja and torch module are need,try turn to 'pip install ninja'.")

    def _suffix(self):
        return "cpp"

    def _text_tail(self, module_name, doc="module Document",
                   functions=("fcc", "bcc", "cubic")) -> str:
        """
        Add following code to f_cpp file, Just for function, rather than class.

        Add following code to f_cpp file.''

        PYBIND11_MODULE(module_name, m) {
        m.doc() = "module Document"; // optional module docstring
        m.def("cubic", &cubic, "Document");
        m.def("fcc", &fcc, "Document");
        m.def("bcc", &bcc, "Document");
        }
        ''


        Parameters
        ----------
        module_name:str
            module name, please make sure the name just contain [a-z][0-9] and _. such as "a", default using the name of f_cpp.
        doc:str
            module document.
        functions:list of str
            function name, the names must be function in cpp file.
        """

        func_str = ['m.def("{i}", &{i}, "function Document"); // functions'.format(i=i) for i in functions]
        func_str = "\n".join(func_str)

        PYBIND11_MODULE = r"""
PYBIND11_MODULE({module_name}, m) {{
m.doc() = "{doc}"; // optional module docstring
{func_str}
}}""".format(module_name=module_name, doc=doc, func_str=func_str)
        return PYBIND11_MODULE

    def quick_import(self, build=False, suffix=".so", with_html=False):
        os.chdir(self.path)
        self.build = build
        from torch.utils import cpp_extension

        def re_build_func_torch():
            mod = cpp_extension.load(
                name=self.module_name,
                sources=[self.file, ],
                verbose=True,
                build_directory=self.path,
                is_python_module=True, )
            return mod

        mod = quick_import(self.module_name, path=None, build=build, suffix=suffix,
                           re_build_func=re_build_func_torch, re_build_func_kwargs={}, log_print=self.log_print)
        os.chdir(self.init_path)
        if self.log_print:
            print("Move back to {}".format(self.init_path))
        return mod


class TorchJitInLine:
    """This type is one JIT of torch, with no setup.py file.

    Examples
    -----------
    source = "

    #include <torch/extension.h>

    #include <iostream>

    torch::Tensor d_sigmoid(torch::Tensor z) {

    auto s = torch::sigmoid(z);

    return (1 - s) * s;}
    "

    >>> from mgetool.draft import TorchJitInLine

    >>> bd = TorchJitInLine(source=source)
    >>> bd.write(functions=["d_sigmoid",])
    >>> a= bd.quick_import(build=True)
    >>> bd.remove()

    """

    def __init__(self, source, module_name="TORCH_EXTENSION_NAME", path=None, temps="temps", warm_start=False,
                 log_print=True
                 ):
        """
        Add one temps dir and copy all the file to this disk to escape pollution.

        Parameters
        ----------
        source:str
            all the cpp source text but not file name.
        module_name:
            name of module
        path:str
            path to build.
        temps: str
            Add one temps dir.
        warm_start:bool
            start from exist file.
        """
        self.log_print = log_print
        self.init_path = os.getcwd()
        # check file
        if module_name == "TORCH_EXTENSION_NAME":
            print("please re set your module name")
        self.source = source

        if path is not None:
            def_pwd(path)

        MODULE_DIR = Path().absolute()
        # temps:
        if temps:
            if warm_start:
                if not os.path.isdir(temps):
                    raise FileNotFoundError("Try to warm start but without no exist {} found".format(temps))
            else:

                if not os.path.isdir(temps):
                    os.mkdir(temps)
                else:
                    warnings.warn("There is exist {temps}. Duplicate files will be overwritten."
                                  "please use remove() to delete temporary file after each test.".format(temps=temps))

            self.temps = temps
            self.path = MODULE_DIR / temps

        else:
            self.temps = None
            self.path = MODULE_DIR
        # check module_name
        self.module_name = module_name
        self.build = True
        self.functions = []
        os.chdir(self.init_path)

    def write(self, functions: list):
        self.functions = functions
        if len(functions) == 0:
            print("function names must be set")

    def quick_import(self, build=False, suffix=".so"):
        os.chdir(self.path)
        self.build = build
        from torch.utils import cpp_extension

        def re_build_func_torch():
            mod = cpp_extension.load_inline(
                name=self.module_name,
                cpp_sources=self.source,
                functions=self.functions,
                verbose=True,
                build_directory=self.path,
                is_python_module=True, )
            return mod

        mod = quick_import(self.module_name, path=self.path, build=build, suffix=suffix,
                           re_build_func=re_build_func_torch, re_build_func_kwargs={}, log_print=self.log_print)
        os.chdir(self.init_path)
        if self.log_print:
            print("Move back to {}".format(self.init_path))
        return mod

    def remove(self, numbers=None):
        """remove files, beyond retrieve, carefully use!!!"""
        os.chdir(self.path)
        if self.temps is None and numbers is None:
            warnings.warn("Difficult to determine which file to delete. please pass the numbers, such as [0,1,2],"
                          "Please confirm the serial number")
            print("Exist file: {}")
            for i in zip(range(len(os.listdir())), os.listdir()):
                print(i)
        elif self.temps is None and numbers is not None:
            files = [os.listdir()[i] for i in numbers]
            for i in files:
                if os.path.isdir(i):
                    print("dir {} is deleted.".format(i))
                    shutil.rmtree(i)
                else:
                    print("file {} is deleted.".format(i))
                    os.remove(i)
            if self.build is False:
                print(
                    "! Please commented out or delete this line in file to prevent mistaken delete "
                    "in next time running\n"
                    "Such as:\n"
                    " >>> # bd.remove([6,7,8,9,10])")
            else:
                print("Delete end")
        else:
            shutil.rmtree(self.path)
            print("Delete end")
        os.chdir(self.init_path)
