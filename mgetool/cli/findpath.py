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
        parser.add_argument('-p', '--path', help='source path',type=str, default=None)
        parser.add_argument('-s', '--suffix', help='suffix of file',type=str,default=None)
        parser.add_argument('-if', '--file_include', help='include file name.',type=str,default=None)
        parser.add_argument('-ef', '--file_exclude', help='exclude file name.',type=str,default=None)
        parser.add_argument('-id', '--dir_include', help='include dir name.',type=str,default=None)
        parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.',type=str,default=None)
        parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=int, default=-1)


    @staticmethod
    def run(parser):
        parser = parser.parse_args()
        bf = BatchFile(parser.path, suffix=parser.suffix)
        bf.filter_dir_name(include=parser.dir_include,exclude=parser.dir_exclude,layer=parser.layer)
        bf.filter_file_name(include=parser.file_include,exclude=parser.file_exclude)
        bf.merge()

        fdir = bf.file_dir
        fdir.sort()
        with open("paths.temp", mode="w") as f:
            f.writelines(fdir)
            os.getcwd()
        print("The paths are stored in '{}'".format(os.getcwd()))




