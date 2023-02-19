# -*- coding: utf-8 -*-

# @Time  : 2022/7/29 17:06
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import os
import re
import shutil
import warnings
from fnmatch import translate
from typing import Union, List

from path import Path

# patten >>>>>
shell_patten_help = r"""
(通配符字符串,使用''包裹,如'*.ext').

*       匹配任意字符.
?       匹配任意单个字符.
[seq]   用来表示一组字符,单独列出：[amk] 匹配 'a','m'或'k'.
[!seq]  不在[]中的字符：[^abc] 匹配除了a,b,c之外的字符. (不建议使用)
"""

# patten >>>>>
re_patten_help = r"""
(通配符字符串,使用''包裹,如'.*.ext')

.       匹配任意字符,除了换行符.
*       (不可单独使用，前面需有字符)
re*     匹配0个或多个的表达式.
re+     匹配1个或多个的表达式.
re?     匹配0个或1个由前面的正则表达式定义的片段,非贪婪方式.
a|b     匹配a或b.
\w      匹配字母数字及下划线.
\W      匹配非字母数字及下划线.
\s      匹配任意空白字符.
\S      匹配任意非空字符.
\d      匹配任意数字,等价于 [0-9].
\D      匹配任意非数字.
[...]   用来表示一组字符,单独列出：[amk] 匹配 'a','m'或'k'.
[^...]  不在[]中的字符：[^abc] 匹配除了a,b,c之外的字符.  (不建议使用)
"""


def shell_to_re_compile_pattern(pat, trans=True, single=True, ):
    """shell to re patten."""
    try:
        if trans:
            res = translate(pat)
            if single is False:
                res = res.replace("?s:", "?m:")
                res = res.replace("\Z", "")
        else:
            res = pat
        return re.compile(res)
    except BaseException as e:
        print("-------------------------\n"
              "1. If trans==False (default) >>> ``re laws``\n"
              "----re module help-------")
        print(re_patten_help)
        print("More: https://docs.python.org/3/library/re.html?highlight=re#module-re\n\n"
              "-------------------------\n"
              "2. If trans==True (-t) >>> ``linux shell laws``\n"
              "---shell module help-----")
        print(shell_patten_help)
        print("------------------------\n")
        print("Wrong patten, check the forward message to change the match patten.\n")
        raise e


class BatchPathMatch:
    def __init__(self, path=".", patten: Union[str]=None, trans=True, abspath=False):
        """

        Parameters
        ----------
        path:str
            total dir of all files.
        patten:str
            match patten.
        trans: bool, default True
            If true, use shell patten to match.
            If False, use re patten to match.
        """
        self.trans = trans

        self.path = Path(path)
        if patten is not None:

            dir_list = self.path.walkdirs()
            patten = self._get_patten(patten)
            self.file_dir = list(self.filter(dir_list, pat=patten, trans=self.trans))

        else:
            self.file_dir = list(self.path.walkdirs())
        if abspath:
            self.file_dir  = self.dir_relpath()

    @staticmethod
    def _get_patten(my_strs):
        if my_strs is not None and " " in my_strs:
            my_strs = str(my_strs).replace("|", " ")
            my_strs = my_strs.split(" ")
            my_strs = [i for i in my_strs if i not in ["", " "]]
        else:
            pass

        if isinstance(my_strs, (tuple, list)):
            if len(my_strs) >= 2:
                my_strs = "|".join(my_strs)
            elif len(my_strs) == 1:
                my_strs = my_strs[0]
            else:
                raise NotImplementedError
        else:
            pass
        return my_strs

    @staticmethod
    def filter(names, pat, trans=True):
        smatch = shell_to_re_compile_pattern(pat, trans=trans).search
        return [ni for ni in names if smatch(ni) is not None]

    def get_leaf_dir(self, dirs=None):
        """Get the leaf dirs. Remove "." path."""

        # The ../paths could be wrong!

        if dirs is None:
            dirs = self.file_dir

        dirs = [i for i in dirs if i not in (".", "./")]
        dirs_text = "\n".join(dirs)
        index = [dirs_text.count(i) - 1 for i in dirs]
        dirs = [dirs[n] for n, i in enumerate(index) if i == 0]
        return dirs

    def dir_relpath(self, path=None):
        """Get the real-path to input path."""
        if path is None:
            path = self.path
        else:
            path = Path(path)

        self.file_dir = [i.relpath(path) for i in self.file_dir]



class BatchFileMatch:
    r"""
    Search files and filter files and re-site files with patten match.

    Examples
    ---------
    >>> bfm = BatchFileMatch(".")
    >>> bfm.filter_dir_name("a")
    >>> bfm.filter_file_name("2")
    >>> print(bfm.file_list)
    >>> bfm.merge()
    >>> print(bfm.file_dir)
    ...

    # copy the file to new path and keep the dir structure
    >>> bfm.copyfile(r"C:\Users\Admin\Desktop\d2")
    # cancel copy
    >>> bfm.copyfile_back()
    ...
    """

    def __init__(self, path=".", suffix=None, patten=None, trans=True):
        """

        Parameters
        ----------
        path:str
            total dir of all files.
        suffix:str
            suffix of file.
            Examples:
                .txt
        patten:str
            match patten.
        trans: bool, default True
            If true, use shell patten to match.
            If False, use re patten to match.
        """
        self.trans = trans

        self.path = Path(path)
        if patten is not None:

            file_list = self.path.walkfiles()
            patten = self._get_patten(patten)
            self.file_list = list(self.filter(file_list, pat=patten, trans=self.trans))

        else:
            if suffix is not None:
                if suffix[:2] == "*.":
                    self.file_list = list(self.path.walkfiles(match=suffix))
                elif suffix[0] != ".":
                    raise SyntaxError("suffix should be start from '.' ")
                else:
                    self.file_list = list(self.path.walkfiles(match=f"*{suffix}"))
            else:
                self.file_list = list(self.path.walkfiles())

        self.file_dir = []
        self._rm_check = 0

    def __repr__(self):
        return f"Root: {self.path}"

    def _comp(self, pat):
        return shell_to_re_compile_pattern(pat, trans=self.trans)

    @staticmethod
    def filter(names, pat, trans=True):
        smatch = shell_to_re_compile_pattern(pat, trans=trans).search
        return [ni for ni in names if smatch(ni) is not None]

    @staticmethod
    def searchcase(name, pat, trans=True):
        smatch = shell_to_re_compile_pattern(pat, trans=trans).search
        return smatch(name) is not None

    @staticmethod
    def matchcase(name, pat, trans=True):
        smatch = shell_to_re_compile_pattern(pat, trans=trans).match
        return smatch(name) is not None

    @staticmethod
    def _get_patten(my_strs):
        if my_strs is not None and " " in my_strs:
            my_strs = str(my_strs).replace("|", " ")
            my_strs = my_strs.split(" ")
            my_strs = [i for i in my_strs if i not in ["", " "]]
        else:
            pass

        if isinstance(my_strs, (tuple, list)):
            if len(my_strs) >= 2:
                my_strs = "|".join(my_strs)
            elif len(my_strs) == 1:
                my_strs = my_strs[0]
            else:
                raise NotImplementedError
        else:
            pass
        return my_strs


    def filter_file_name_parent_folder(self, exclude=None):
        """
        filter the dir and sub-files contain the file!

        Parameters
        ----------
        exclude: str,list
            Delete the filename with exclude str
            such as hold "ast_cap" file and delete "ast_tep" with "tep" str.

        """

        if exclude:
            files = [i.name for i in self.file_list]
            exclude = self._get_patten(exclude)
            if not any([i in exclude for i in ["*", "|", "?", ".", "\\", "[", "+"]]):  # 简单模式
                del_dirs = [self.file_list[r].parent for r, i in enumerate(files) if exclude in i]
            else:
                pt2 = self._comp(exclude)
                del_dirs = [self.file_list[r].parent for r, i in enumerate(files) if re.search(pt2, i) is not None]

            file_dir = list((set([i.parent for i in self.file_list])))
            file_dir = self.get_leaf_dir(file_dir)

            keep_dirs = set(file_dir) - set(del_dirs)
            fs = []
            [fs.extend(ki.files()) for ki in keep_dirs]
            self.file_list = fs
            self.file_dir = list(keep_dirs)

    def filter_file_name(self, include: Union[List[str], str] = None, exclude: Union[List[str], str] = None,
                         patten: str = None):
        """
        Filter file name.
        ``exclude`` for in ``filter_file_name`` are just suitable for filter one file.
        And don't filter the other files in same dir! Thus, the dirs are keep exist due to other files!!!
        use ``filter_file_name_parent_folder`` instead.

        Parameters
        ----------
        include:str,list
            Get the filename with include str
            such as hold "ast_tep" file with "ast" string
        exclude: str,list
            Delete the filename with exclude str
            such as hold "ast_cap" file and delete "ast_tep" with "tep" str.
        patten: str
            Patten for match, such as ‘*’,‘^’,'.' in ‘re’ module.
        """

        if patten:
            patten = self._get_patten(patten)
            pt = self._comp(patten)
            self.file_list = [i for i in self.file_list if re.search(pt, i) is not None]
        else:
            if not include and not exclude:
                pass
            else:
                # 只要出现include内容 - 保留,或者只要出现exclude内容 - 删除
                files = [i.name for i in self.file_list]

                if include:
                    include = self._get_patten(include)
                    if not any([i in include for i in ["*", "|", "?", ".", "\\", "[", "+"]]):  # 简单模式
                        self.file_list = [self.file_list[r] for r, i in enumerate(files) if include in i]
                        files = [i for i in files if include in i] if exclude else []
                    else:
                        pt = self._comp(include)
                        index = [r for r, i in enumerate(files) if re.search(pt, i) is not None]
                        self.file_list = [self.file_list[r] for r in index]
                        files = [files[r] for r in index] if exclude else []

                if exclude:
                    exclude = self._get_patten(exclude)
                    if not any([i in exclude for i in ["*", "|", "?", ".", "\\", "[", "+"]]):  # 简单模式
                        self.file_list = [self.file_list[r] for r, i in enumerate(files) if exclude not in i]
                    else:
                        pt2 = self._comp(exclude)
                        self.file_list = [self.file_list[r] for r, i in enumerate(files) if re.search(pt2, i) is None]

    def filter_dir_name(self, include: Union[List[str], str] = None, exclude: Union[List[str], str] = None,
                        layer: Union[None, List, int, str] = -1, patten: str = None):
        """
        Filter the dir (and its sub_file).

        Parameters
        ----------
        include:str,list of str
            Get the filename with include str
            such as hold "ast_tep" with "ast" string
        exclude: str, list of str
            Delete the filename with exclude str.
            such as hold "ast_cap" and delete "ast_tep" with "tep" str.
        layer:int,list
            If list, check the name of these layers.
            Filter dir with target layer, all the dir should contain the sublayer!
            Examples:
                for /home,
                /home/ast, -3
                /home/ast/eag, -2
                /home/ast/eag/kgg, -1
        patten: str
            Patten for match, such as ‘*’,‘^’,'.' in ‘re’ module.
        """

        if isinstance(layer, str):
            if " " in layer:
                layer = str(layer).split(" ")
                layer = [int(i) for i in layer]
                if len(layer) == 1:
                    layer = layer[0]
                    layer = int(layer)
            else:
                layer = int(layer)
                if layer == 0:
                    layer = None

        if patten:
            patten = self._get_patten(patten)
            pt = self._comp(patten)
            self.file_list = [i for i in self.file_list if re.search(pt, i) is not None]
        else:
            if not include and not exclude:
                pass
            else:
                try:

                    if layer is None or layer == 0:  # 全目录匹配
                        file_dir = [i.parent for i in self.file_list]
                    elif layer == -1:  # 匹配最后一层目录
                        file_dir = [i.parent.name for i in self.file_list]
                    elif isinstance(layer, int):  # 匹配单层目录
                        file_dir = [i.parent.splitall()[layer] for i in self.file_list]
                    elif isinstance(layer, (tuple, list)):  # 匹配多层目录
                        file_dir = [i.parent.splitall() for i in self.file_list]
                        file_dir = ["/".join([i[ll] for ll in layer]) for i in file_dir]
                    else:
                        raise NotImplementedError("Wrong type of 'layer'.")
                except IndexError as e:
                    print(f"----> Make sure all the sub-dirs with in depth {layer}. Too big for 'layer'!!!")
                    raise e

                # 只要出现include内容 - 保留,或者只要出现exclude内容 - 删除
                if include:
                    include = self._get_patten(include)
                    if not any([i in include for i in ["*", "|", "?", ".", "\\", "[", "+"]]):  # 简单模式
                        self.file_list = [self.file_list[r] for r, i in enumerate(file_dir) if include in i]
                        file_dir = [i for i in file_dir if include in i] if exclude else []
                    else:
                        pt = self._comp(include)
                        index = [r for r, i in enumerate(file_dir) if re.search(pt, i) is not None]
                        self.file_list = [self.file_list[r] for r in index]
                        file_dir = [file_dir[r] for r in index] if exclude else []

                if exclude:
                    exclude = self._get_patten(exclude)
                    if not any([i in exclude for i in ["*", "|", "?", ".", "\\", "[", "+"]]):  # 简单模式
                        self.file_list = [self.file_list[r] for r, i in enumerate(file_dir) if exclude not in i]
                    else:
                        pt2 = self._comp(exclude)
                        self.file_list = [self.file_list[r] for r, i in enumerate(file_dir) if
                                          re.search(pt2, i) is None]

    def merge(self, abspath=False, force_relpath=False):
        """Merge dir and file name together, Get dir names."""

        if abspath:
            self.file_list = [i.abspath() for i in self.file_list]
        else:
            if force_relpath:
                self.file_list = [i.relpath(".") for i in self.file_list]
                # add "./"
                self.file_list = [Path.joinpath(".", i) if i[0] not in ["." or "/"] else i for i in self.file_list]
            else:
                pass
        file_dir = list((set([i.parent for i in self.file_list])))
        file_dir.sort()
        self.file_dir = file_dir
        return self.file_list

    def relpathto(self, path=None):
        """Get the real-path to input path."""
        if path is None:
            path = self.path
        else:
            path = Path(path)

        return [i.relpath(path) for i in self.file_list]

    @staticmethod
    def _copy_user(i, j):
        if os.path.isdir(i):
            shutil.copytree(i, j)
        else:
            path_i = os.path.split(j)[0]
            if not os.path.exists(path_i):
                os.makedirs(path_i)
            shutil.copy(i, j)

    @staticmethod
    def _move_user(i, j):
        if os.path.isdir(i):
            shutil.move(i, j)
        else:
            path_i = os.path.split(j)[0]
            if not os.path.exists(path_i):
                os.makedirs(path_i)
            shutil.move(i, j)

    def copyfile(self, newpath):
        """Copy files to newpath and keep the dir tree."""
        file_list = self.relpathto(self.path)
        newpath = Path(newpath)
        self._file_list_new = [Path.joinpath(newpath, i) for i in file_list]
        self._file_list_old = [Path.joinpath(self.path, i) for i in file_list]

        [self._copy_user(i, j) for i, j in zip(self._file_list_old, self._file_list_new)]
        print("If you want cancel the 'copyfile', please use 'copyfile_back' right away, "
              "The new files would be deleted.")

    def removedirs_p(self, dirs=None):
        """Remove dirs."""
        if dirs is None:
            dirs = self.file_dir
        [i.removedirs_p() for i in dirs]

    def copyfile_back(self, del_dir=True):
        """Just revert for 'copyfile' after 'copyfile'."""
        [i.remove_p() for i in self._file_list_new]
        if del_dir:
            file_dir = list((set([i.parent for i in self._file_list_new])))
            self.removedirs_p(file_dir)

    to_path = copyfile

    def movefile(self, newpath):
        """Move files to newpath and keep the dir tree."""
        file_list = self.relpathto(self.path)
        newpath = Path(newpath)
        self._file_list_new = [Path.joinpath(newpath, i) for i in file_list]
        self._file_list_old = [Path.joinpath(self.path, i) for i in file_list]
        [self._move_user(i, j) for i, j in zip(self._file_list_old, self._file_list_new)]
        print("If you want cancel the 'movefile', please use 'movefile_back' right away.")
        file_dir = list((set([i.parent for i in self._file_list_old])))
        self.removedirs_p(file_dir)

    def movefile_back(self, del_dir=True):
        """Just revert for 'movefile' after 'movefile'."""
        [self._move_user(j, i) for i, j in zip(self._file_list_old, self._file_list_new)]
        if del_dir:
            file_dir = list((set([i.parent for i in self._file_list_new])))
            self.removedirs_p(file_dir)

    def rmfile(self, del_dir=True):
        """Remove are dangerous, Make sure what are you doing!!！"""
        if len(list(self.file_list)) == 0:
            print("Empty file_list.")
        else:
            if self._rm_check > 2:
                [i.remove_p() for i in self.file_list]
                if del_dir:
                    file_dir = list((set([i.parent for i in self.file_list])))
                    self.removedirs_p(file_dir)
            elif self._rm_check == 2:
                warnings.warn("It would delete all the files in disk (containing in file_list) the next time!!!\n")
                self._rm_check += 1
            else:
                warnings.warn(
                    f"{self.file_list}\n"
                    "It would delete all the files in disk (containing in file_list), and are irreversible !!!\n"
                    "> Check the 'file_list' above, make sure it does is what you want delete carefully!\n"
                    "If you completely determine delete it, typing 'xxx.rmfile()' again in interactive window.\n"
                    f"Until the rm_check==3. (Now: rm_check={self._rm_check}).")
                self._rm_check += 1

    def get_leaf_dir(self, dirs=None):
        """Get the leaf dirs. Remove "." path."""

        # The ../paths could be wrong!

        if dirs is None:
            dirs = self.file_dir

        dirs = [i for i in dirs if i not in (".", "./")]
        dirs_text = "\n".join(dirs)
        index = [dirs_text.count(i) - 1 for i in dirs]
        dirs = [dirs[n] for n, i in enumerate(index) if i == 0]
        return dirs

# if __name__=="__main__":
#     bfm = BatchFileMatch(r"C:\Users\Administrator\PycharmProjects\samples\Instance\Instance_mo2co2\MoCMo-O-4")
#
#     # bfm.filter_dir_name(exclude="Re|Co", include="*ure_static",layer=0)
#     bfm.filter_file_name_parent_folder(exclude="*.lobster")
#
#     print(bfm.file_list)
#     bfm.merge()
#     print(bfm.file_dir)
