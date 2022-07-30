# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 14:29
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os

from mgetool.imports import BatchFileMatch


def run(args, parser):
    """
    # simple_patten >>>>>
    *       匹配任意字符
    ?       匹配任意单个字符
    [seq]   用来表示一组字符,单独列出：[amk] 匹配 'a'，'m'或'k'
    [!seq]  不在[]中的字符：[^abc] 匹配除了a,b,c之外的字符。
    """

    bf = BatchFileMatch(args.path, suffix=args.suffix, simple_patten=args.match_patten)

    if " " in args.layer:
        layer = str(args.layer).split(" ")
        layer = [int(i) for i in layer]
        if len(layer)==1:
            layer = layer[0]
    else:
        layer = int(args.layer)
        if layer == 0:
            layer = None

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
    os.chdir(args.path)


    with open(str(args.store_name), mode="w") as f:
        fdir = "\n".join(fdir)
        f.writelines(fdir)

    print("The paths '{}' are stored in '{}'.".format(args.store_name, str(os.getcwd())))
    print("OK")


class CLICommand:
    """寻找符合要求所有的子路径, 查看参数帮助使用 -h。

    如果在 mgetool 中运行:


    Example:

    如果有字符串名字使用任何的 *，$ 等匹配字符，请使用 ’‘ 包裹字符串，否则可以省略。

        $ mgetool findpath -if '.*.xml'

        $ mgetool findpath -p /home/dir_name -if POSCAR

        $ mgetool findpath  -id my_dir1 -ed my_dir2

    使用空格划分

        $ mgetool findpath -id ini_opt  -l '-3 -2 -1'

        $ mgetool findpath -id 'ini_opt ini_static'

    如果有字符串名字使用任何的 *，$ 等匹配字符，请使用 ’‘ 包裹字符串。

        $ mgetool findpath -if '.*.xml'

    如果复制该脚本到某处，仅运行单个脚本:

    Example:

        $ python findpath.py -p /home/dir_name -if POSCAR

    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-m', '--match_patten', help='match_patten', type=str, default=None)
        parser.add_argument('-p', '--path', help='source path', type=str, default=".")
        parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
        parser.add_argument('-if', '--file_include', help='include file name.', type=str, default=None)
        parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
        parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
        parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
        parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=str, default="-1")
        parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=False)
        parser.add_argument('-o', '--store_name', help='out file name, default paths.temp', type=str,
                            default="paths.temp")

    @staticmethod
    def run(args, parser):
        run(args, parser)


if __name__ == '__main__':
    # 命令行模式
    import argparse

    parser = argparse.ArgumentParser(description="根据路径文件，寻找所有符合要求的文件子路径："
                                                 "python findpath.py -p /home/dir_name -if POSCAR"
                                     )

    parser.add_argument('-m', '--match_patten', help='match_patten, just support "*". ', type=str, default=None, )
    parser.add_argument('-p', '--path', help='source path', type=str, default=".")
    parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
    parser.add_argument('-if', '--file_include', help='include file name.', type=str, default=None)
    parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
    parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
    parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
    parser.add_argument('-l', '--layer', help='dir depth, default the last layer', type=str, default="-1")
    parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=False)
    parser.add_argument('-o', '--store_name', help='out file name, default paths.temp', type=str, default="paths.temp")
    args = parser.parse_args()
    run(args, parser)
