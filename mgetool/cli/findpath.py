# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 14:29
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os
import pathlib

from mgetool.imports import BatchFileMatch
from mgetool.imports.batchfilematch import re_patten_help, shell_patten_help, BatchPathMatch


_dos_help = f"""寻找符合要求所有的叶节点路径.

运行方式: (1) findpath ... (2) mt findpath ... (3) mgetool findpath ...

Key 1. 如果有字符串名字使用任何的 * $ 等匹配字符, 请使用单引号 '' 包裹字符串, 否则引号可以省略.

    (若不包裹, 通配字符首先由系统shell解释, 再传递到python, 请确定您知悉自己的操作目的再作决定.)

    S1. 匹配包含任意xml文件的路径:
    $ findpath -if '*.xml'
    
    S2. 匹配/home/dir_name下, 包含POSCAR文件的子路径:
    $ findpath -p /home/dir_name -if POSCAR
    
    S3. 匹配路径名称包含my_dir1,不包含my_dir2路径, -l 0 代表全路径（不包含文件名）:
    $ findpath -id my_dir1 -ed my_dir2 -l 0
    
Key 2. 默认参数采用全路径（包含文件名）匹配, 与传统shell一致.

    S4. 匹配包含任意xml文件的路径:
    $ findpath './*/*/POSCAR'
    
Key 3. 通配符选择使用解析方式 linux shell 默认为方式, 可使用 -t 切换为 python re 模块解析.
    需要对 python re 模块有一定的了解.

    Shell Patten >>>
    {shell_patten_help}
    Re    Patten （with -t） >>>
    {re_patten_help}
    Patten       <<<

Key 4. 多重可选匹配使用 | 或者空格划分.

    S5. 匹配包含倒数三层文件夹,出现ini_opt名称的路径:
    $ findpath -id ini_opt  -l '-3 -2 -1'

    S6. 匹配(默认最后一层文件夹)出现ini_opt或者ini_static的路径:
    $ findpath -id 'ini_opt|ini_static' -t
    
注1：后续可使用命令: makebatch 创建批处理脚本, 并自定义处理命令.
注2：复杂功能实现, 请python脚本：

    >>> from mgetool.imports import BatchFileMatch
    >>> bf = BatchFileMatch()
    >>> ...
"""


def run(args, parser):
    print("\nCollecting all Paths ...")
    if args.suffix is None and\
       args.dir_include is None and args.dir_exclude is None and args.file_include is None and\
       args.file_exclude is None:
        print("Simple version...")

        if args.match_patten_arg is None:
            bf = BatchPathMatch(args.path, patten=args.match_patten, trans=args.translate,
                                abspath=args.abspath,relpath=args.relpath)
        else:
            bf = BatchPathMatch(args.path, patten=args.match_patten_arg, trans=args.translate,
                                abspath=args.abspath,relpath=args.relpath)

    else:

        # situation 1
        # if the 'match_patten_arg' is use [^...] or [!seq] just for match file name,
        # This would find all file matched with patten in the dirs,
        # The dirs would remain due to the file. thus the  [^...] or [!seq] (for file name) would not filter the dirs.
        if args.match_patten_arg is None:
            bf = BatchFileMatch(args.path, suffix=args.suffix, patten=args.match_patten, trans=args.translate)
        else:
            bf = BatchFileMatch(args.path, suffix=args.suffix, patten=args.match_patten_arg, trans=args.translate)

        print("Filter the Paths ...")

        # situation 2
        # (the parent of parent dir or more top-lever could be residual. if dir_exclude not None)
        bf.filter_dir_name(include=args.dir_include, exclude=args.dir_exclude, layer=args.layer)

        # situation 1
        # if use 'exclude' in this function, the dirs containing exclude file would remain, due to the other file in dirs.
        # thus, exclude are set to next function.
        bf.filter_file_name(include=args.file_include)

        # this is the real, to delete the dirs containing exclude file.
        bf.filter_file_name_parent_folder(exclude=args.file_exclude)

        if args.dir_exclude is not None:
            print("Use '-ed' could result to parent folder residue. Manual check and delete is recommended.")

        bf.merge(abspath=args.abspath,relpath=args.path, force_relpath=args.relpath)
        


    if not args.parent:
        fdir = bf.get_leaf_dir()
    else:
        fdir = bf.file_dir

    if args.reverse:
        fdir.reverse()

    num = len(fdir)
    
    fdir = [str(i) for i in fdir]

    if args.not_print is True:
        print("\nPaths:\n[")
        [print(i) for i in fdir]
        print("]\n")

    print("Write Out File ...")

    os.chdir(args.path)
    
    pathlib.Path(args.store_name).parent.mkdir(parents=True, exist_ok=True)

    with open(str(args.store_name), mode="w") as f:
        fdir = "\n".join(fdir)
        f.writelines(fdir)

    print("The '{}' with {} paths are stored in '{}'.".format(args.store_name, num, str(os.getcwd())))
    print("Done.\n")


class CLICommand:
    __doc__ = _dos_help

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(dest='match_patten_arg', nargs="?", help='match_patten.', type=str, default=None)
        parser.add_argument('-m', '--match_patten', help='match_patten.', type=str, default=None)
        parser.add_argument('-p', '--path', help='source path.', type=str, default=".")
        parser.add_argument('-s', '--suffix', help='suffix of file.', type=str, default=None)
        parser.add_argument('-if', '--file_include', help='include file name.', type=str, default=None)
        parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
        parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
        parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
        parser.add_argument('-t', '--translate', help='If not use, offer shell patten, If use, offer python re patten to match.',
                            action="store_false")
        parser.add_argument('-l', '--layer', help='dir depth, default the last layer.', type=str, default="-1")
        parser.add_argument('-abs', '--abspath', help='return abspath.', action="store_true")
        parser.add_argument('-rel', '--relpath', help='return related path.', action="store_true")
        parser.add_argument('-o', '--store_name', help='out file name, default paths.temp.', type=str,
                            default="paths.temp")
        parser.add_argument('-np', '--not_print', help='not print.', action="store_false")
        parser.add_argument('-parent', '--parent', help='with parent or not.', action="store_true")
        parser.add_argument('-r', '--reverse', help='reverse the list.', action="store_true")

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()

    @staticmethod
    def run(args, parser):
        run(args, parser)


def main():
    """
    Example:
        $ python this.py -p /home/dir_name
        $ python this.py -f /home/dir_name/path.temp
    """

    from mgetool.cli._formatter import Formatter
    import argparse

    parser = argparse.ArgumentParser(description=_dos_help, formatter_class=Formatter)
    CLICommand.add_arguments(parser=parser)
    args = CLICommand.parse_args(parser=parser)
    CLICommand.run(args=args, parser=parser)


if __name__ == '__main__':
    main()
