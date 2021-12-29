# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 14:29
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os

from mgetool.imports import BatchFile


def run(args, parser):
    # args = args.parse_args()
    bf = BatchFile(args.path, suffix=args.suffix)
    bf.filter_dir_name(include=args.dir_include, exclude=args.dir_exclude, layer=args.layer)
    bf.filter_file_name(include=args.file_include, exclude=args.file_exclude)
    bf.merge()

    fdir = bf.file_dir
    fdir.sort()
    os.chdir(args.path)
    if not args.abspath:
        absp = os.path.abspath(args.path)
        fdir = [i.replace(absp, ".") for i in fdir]
    with open(str(args.store_name), mode="w") as f:
        fdir = "\n".join(fdir)
        f.writelines(fdir)

    print("The paths '{}' are stored in '{}'.".format(args.store_name,str(os.getcwd())))
    print("OK")


class CLICommand:
    """
    寻找符合要求所有的子路径, 查看参数帮助使用 -h。

    如果在 mgetool 中运行:

    Example:

        $ mgetool findpath -p /home/dir_name -if POSCAR

    如果复制该脚本到某处，仅运行单个脚本:

    Example:

        $ python findpath.py -p /home/dir_name -if POSCAR
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path', help='source path', type=str, default=".")
        parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
        parser.add_argument('-if', '--file_include', help='include file name.', type=str, default=None)
        parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
        parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
        parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
        parser.add_argument('-l', '--layer', help='dir depth, default the last layer', type=int, default=-1)
        parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=False)
        parser.add_argument('-o', '--store_name', help='out file name, default paths.temp', type=str, default="paths.temp")

    @staticmethod
    def run(args, parser):
        run(args,parser)


if __name__ == '__main__':
    # 命令行模式
    import argparse

    parser = argparse.ArgumentParser(description="根据路径文件，寻找所有符合要求的文件子路径："
                                                 "python findpath.py -p /home/dir_name -if POSCAR"
                                                 )

    parser.add_argument('-p', '--path', help='source path', type=str, default=".")
    parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
    parser.add_argument('-if', '--file_include', help='include file name.', type=str, default=None)
    parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
    parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
    parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
    parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=int, default=-1)
    parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=False)
    parser.add_argument('-o', '--store_name', help='out file name,default paths.temp', type=str, default="paths.temp")
    args = parser.parse_args()
    run(args, parser)
