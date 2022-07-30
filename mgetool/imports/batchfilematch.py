# -*- coding: utf-8 -*-

# @Time  : 2022/7/29 17:06
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import os
import re
import shutil
import warnings
from typing import Union, List

from path import Path

# simple_patten >>>>> （for simple_match）
shell_patten_help = """
*       匹配任意字符
?       匹配任意单个字符
[seq]   用来表示一组字符,单独列出：[amk] 匹配 'a'，'m'或'k'
[!seq]  不在[]中的字符：[^abc] 匹配除了a,b,c之外的字符。
"""


# patten >>>>> （for include,exclude,match）
re_patten_help = """
# .	匹配任意字符，除了换行符
# re*	匹配0个或多个的表达式。
# re+	匹配1个或多个的表达式。
# re?	匹配0个或1个由前面的正则表达式定义的片段，非贪婪方式
# a| b	匹配a或b
# [...]	用来表示一组字符,单独列出：[amk] 匹配 'a'，'m'或'k'
# [^...]	不在[]中的字符：[^abc] 匹配除了a,b,c之外的字符。
# \w	匹配字母数字及下划线
# \W	匹配非字母数字及下划线
# \s	匹配任意空白字符，等价于 [ \t\n\r\f]。
# \S	匹配任意非空字符
# \d	匹配任意数字，等价于 [0-9].
# \D	匹配任意非数字
"""


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
    #copy the file to new path and keep the dir structure
    >>> bfm.copyfile(r"C:\Users\Admin\Desktop\d2")
    #cancel copy
    >>> bfm.copyfile_back()
    ...
    """

    def __init__(self, path=".", suffix=None, simple_patten=None):
        """

        Parameters
        ----------
        path:str
            total dir of all files.
        suffix:str
            suffix of file.
            Examples:
                .txt
        simple_patten:str
            simple patten for match just following:
            *       matches everything
            ?       matches any single character
            [seq]   matches any character in seq
            [!seq]  matches any char not in seq
        """

        self.path = Path(path)
        if simple_patten is not None:

            self.file_list = list(self.path.walkfiles(match=simple_patten))

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

    @staticmethod
    def _get_patten(my_strs):
        if isinstance(my_strs, (tuple, list)):
            if len(my_strs) >= 2:
                my_strs = "|".join(my_strs)
            elif len(my_strs) == 1:
                my_strs = my_strs[0]
            else:
                raise NotImplementedError
        elif isinstance(my_strs, str):
            pass
        else:
            pass
        return my_strs

    def _comp(self, patten):
        try:
            return re.compile(patten)
        except BaseException as e:

            print("-------------------------\n"
            "1. '-id','-ed','if','ef', 'include', 'exclude' obey >>> ``re laws``\n"
            "----re module help-------")
            print(re_patten_help)
            print("More: https://docs.python.org/3/library/re.html?highlight=re#module-re\n\n"
            "-------------------------\n"
            "2. '-m','simple_patten'                        obey >>> ``linux shell laws``\n"
            "---shell module help-----")
            print(shell_patten_help)
            print("------------------------\n")
            raise e

    def match(self, patten: str = None):
        """
        More powerful match in 're'.
        
        Parameters
        ----------
        patten: str
            Patten for match, such as ‘*’,‘^’,'.' in ‘re’ module.
        """
        pt = self._comp(patten)
        self.file_list = [i for i in self.file_list if re.search(pt, i) is not None]

    def filter_file_name(self, include: Union[List[str], str] = None, exclude: Union[List[str], str] = None,
                         patten: str = None):
        """

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
            pt = self._comp(patten)
            self.file_list = [i for i in self.file_list if re.search(pt, i) is not None]
        else:
            # 只要出现include内容 - 保留,或者只要出现exclude内容 - 删除
            files = [i.name for i in self.file_list]
            include = self._get_patten(include)
            if include:
                if "*" not in include and "|" not in include and "?" not in include:  # 简单模式
                    self.file_list = [self.file_list[r] for r, i in enumerate(files) if include in i]
                else:
                    pt = self._comp(include)
                    index = [r for r, i in enumerate(files) if re.search(pt, i) is not None]
                    self.file_list = [self.file_list[r] for r in index]

            exclude = self._get_patten(exclude)
            if exclude:
                if "*" not in exclude and "|" not in exclude and "?" not in exclude:  # 简单模式
                    self.file_list = [self.file_list[r] for r, i in enumerate(files) if exclude not in i]
                else:
                    pt2 = self._comp(exclude)
                    index = [r for r, i in enumerate(files) if re.search(pt2, i) is None]
                    self.file_list = [self.file_list[r] for r in index]

    def filter_dir_name(self, include: Union[List[str], str] = None, exclude: Union[List[str], str] = None,
                        layer: Union[None, List, int] = -1, patten: str = None):
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
        if patten:
            pt = self._comp(patten)
            self.file_list = [i for i in self.file_list if re.search(pt, i) is not None]
        else:
            try:
                file_dir = [i.parent for i in self.file_list]
                if layer is None or layer == 0:  # 全目录匹配
                    pass
                elif layer == -1:  # 匹配最后一层目录
                    file_dir = [i.name for i in file_dir]
                elif isinstance(layer, int):  # 匹配单层目录
                    file_dir = [i.splitall()[layer] for i in file_dir]
                elif isinstance(layer, (tuple, list)):  # 匹配多层目录
                    file_dir = [i.splitall() for i in file_dir]
                    file_dir = ["/".join([i[ll] for ll in layer]) for i in file_dir]
                else:
                    raise NotImplementedError("Wrong type of 'layer'.")
            except IndexError as e:
                print(f"> Make sure all the sub-dirs with in depth {layer}. Too big for 'layer'!!!")
                raise e

            # 只要出现include内容 - 保留,或者只要出现exclude内容 - 删除
            include = self._get_patten(include)
            if include:
                if "*" not in include and "|" not in include and "?" not in include and "/" not in include:  # 简单模式
                    self.file_list = [self.file_list[r] for r, i in enumerate(file_dir) if include in i]
                else:
                    pt = self._comp(include)
                    index = [r for r, i in enumerate(file_dir) if re.search(pt, i) is not None]
                    self.file_list = [self.file_list[r] for r in index]
            exclude = self._get_patten(exclude)
            if exclude:
                if "*" not in exclude and "|" not in exclude and "?" not in exclude and "/" not in exclude:  # 简单模式
                    self.file_list = [self.file_list[r] for r, i in enumerate(file_dir) if exclude not in i]
                else:
                    pt2 = self._comp(exclude)
                    index = [r for r, i in enumerate(file_dir) if re.search(pt2, i) is None]
                    self.file_list = [self.file_list[r] for r in index]

    def merge(self, abspath=False):
        """Merge dir and file name together, Get dir names."""
        if abspath:
            self.file_list = [i.abspath() for i in self.file_list]
        else:
            self.file_list = [i.relpath(".") for i in self.file_list]
            # add "./"
            self.file_list = [Path.joinpath(".", i) if i[0] not in ["." or "/"] else i for i in self.file_list ]
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
        """Get the leaf dirs."""

        if dirs is None:
            dirs = self.file_dir

        dirs = ["$WORKPATH" if i == "." else i + '/' for i in dirs]
        dirs_text = "\n".join(dirs)
        index = [dirs_text.count(i) - 1 for i in dirs]
        dirs =  [dirs[n] for n,i in enumerate(index) if i == 0]
        return [Path(".") if i == "$WORKPATH" else i for i in dirs]

