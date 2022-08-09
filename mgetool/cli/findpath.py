# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 14:29
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os
from mgetool.imports import BatchFileMatch
from mgetool.imports.batchfilematch import re_patten_help, shell_patten_help

_dos_help = f"""寻找符合要求所有的叶节点路径, 查看参数帮助使用 -h.

运行方式: (1) findpath ... (2) mt findpath ... (3) mgetool findpath ...

Key 1. 如果有字符串名字使用任何的 *,$ 等匹配字符,请使用单引号 ’‘ 包裹字符串,否则可以省略.
    （若不包裹,通配字符首先由系统shell解释,再传递到python, 请确定您知悉自己的操作目的再作决定.）

    匹配包含任意xml文件的路径
    $ findpath -if '*.xml'
    
    匹配/home/dir_name下,包含POSCAR文件的子路径
    $ findpath -p /home/dir_name -if POSCAR
    
    匹配路径名称包含my_dir1,不包含my_dir2路径, -l 0 代表全路径（不包含文件名）
    $ findpath -id my_dir1 -ed my_dir2 -l 0
    
Key 2. -m 参数采用全路径（包含文件名）匹配,与传统shell一致.

    匹配包含任意xml文件的路径
    $ findpath -m './*/*/POSCAR'
    
Key 3. 通配符选择使用解析方式默认为linux shell 方式,可使用 -t False 切换为 python re 模块解析.
    其功能更加强大复杂,需要对python re 模块有一定的了解.

    # Shell Patten >>>
    {shell_patten_help}
    # Re    Patten >>>
    {re_patten_help}
    # Patten       <<<

Key 4. 多重可选匹配使用 | 或者空格划分.

    匹配包含倒数三层文件夹,出现ini_opt名称的路径.
    $ findpath -id ini_opt  -l '-3 -2 -1'

    匹配（默认最后一层文件夹）出现ini_opt或者ini_static的路径
    $ findpath -id 'ini_opt|ini_static'
    
注1：后续可使用命令：makebatch 创建批处理脚本,并自定义.
注2：复杂功能实现,请使用python交互模式或python脚本：

    >>> from mgetool.imports import BatchFileMatch
    >>> bf = BatchFileMatch()
    >>> ...
"""


def run(args, parser):
    print("Collecting all Paths ...")

    bf = BatchFileMatch(args.path, suffix=args.suffix, patten=args.match_patten, trans=args.translate)

    if " " in args.layer:
        layer = str(args.layer).split(" ")
        layer = [int(i) for i in layer]
        if len(layer) == 1:
            layer = layer[0]
    else:
        layer = int(args.layer)
        if layer == 0:
            layer = None

    print("Filter the Paths ...")

    if args.dir_include is not None and " " in args.dir_include:
        dir_include = str(args.dir_include).split(" ")
    else:
        dir_include = args.dir_include

    if args.dir_exclude is not None and " " in args.dir_exclude:
        dir_exclude = str(args.dir_exclude).split(" ")
    else:
        dir_exclude = args.dir_exclude

    bf.filter_dir_name(include=dir_include, exclude=dir_exclude, layer=layer)

    bf.filter_file_name(include=args.file_include, exclude=args.file_exclude)

    bf.merge(abspath=args.abspath)

    fdir = bf.get_leaf_dir()
    num = len(fdir)

    print("Write Out File ...")

    os.chdir(args.path)

    with open(str(args.store_name), mode="w") as f:
        fdir = "\n".join(fdir)
        f.writelines(fdir)

    print("The '{}' of {} paths are stored in '{}'.".format(args.store_name, num,  str(os.getcwd())))
    print("OK")


class CLICommand:
    __doc__ = _dos_help

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-m', '--match_patten', help='match_patten.', type=str, default=None)
        parser.add_argument('-p', '--path', help='source path.', type=str, default=".")
        parser.add_argument('-s', '--suffix', help='suffix of file.', type=str, default=None)
        parser.add_argument('-if', '--file_include', help='include file name.', type=str, default=None)
        parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
        parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
        parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
        parser.add_argument('-t', '--translate', help='If True, use shell patten, If False, use re patten to match.',
                            type=bool, default=True)
        parser.add_argument('-l', '--layer', help='dir depth,default the last layer.', type=str, default="-1")
        parser.add_argument('-abspath', '--abspath', help='return abspath.', type=bool, default=False)
        parser.add_argument('-o', '--store_name', help='out file name, default paths.temp.', type=str,
                            default="paths.temp")

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
