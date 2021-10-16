# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 14:29
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os

from mgetool.imports import BatchFile


class CLICommand:

    """
    寻找符合要求的路径。

    Example:

        $ mgetool findpath -p /home/dir_name -if POSCAR
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path', help='source path',type=str, default=".")
        parser.add_argument('-s', '--suffix', help='suffix of file',type=str,default=None)
        parser.add_argument('-if', '--file_include', help='include file name.',type=str,default=None)
        parser.add_argument('-ef', '--file_exclude', help='exclude file name.',type=str,default=None)
        parser.add_argument('-id', '--dir_include', help='include dir name.',type=str,default=None)
        parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.',type=str,default=None)
        parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=int, default=-1)
        parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=False)

    @staticmethod
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
            fdir = [i.replace(absp,".") for i in fdir]
        with open("paths.temp", mode="w") as f:
            fdir = "\n".join(fdir)
            f.writelines(fdir)

        os.getcwd()
        print("The paths are stored in '{}'".format(os.getcwd()))




