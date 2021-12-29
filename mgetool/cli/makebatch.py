# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 15:23
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os
import pathlib


def trans_str(string):
    lines = string.split("\\n")
    print(lines)
    lines = "\n".join(lines)
    return lines


def make_batch(paths, cmd="cd $i \necho i \ncd ..", out_file_name="batch.sh"):
    batch_str = """#!/bin/bash
echo $dirname

old_path = $dirname

do

for i in {}

{}

cd old_path

done
        """.format(paths, trans_str(cmd))

    batch_str = batch_str.replace("'", "")
    batch_str = batch_str.replace("[", "")
    batch_str = batch_str.replace("]", "")
    batch_str = batch_str.replace(",", "")
    dirs = os.path.join(os.path.expanduser("~"),"tmp")
    if not os.path.isdir(dirs):
        os.mkdir(dirs)
    os.chdir(dirs)

    bach = open(out_file_name, "w")
    bach.write(batch_str)
    bach.close()
    print("The batch file '{}' is stored in '{}'".format(out_file_name, os.getcwd()))


def make_batch_from_file(path_file, cmd="cd $i \necho i \ncd ..", out_file_name="batch.sh"):
    batch_str = """#!/bin/bash
echo $dirname

old_path = $dirname

for i in $(cat {})

do

{}

cd old_path

done
    """.format(path_file, trans_str(cmd))

    batch_str = batch_str.replace("'", "")
    batch_str = batch_str.replace("[", "")
    batch_str = batch_str.replace("]", "")
    batch_str = batch_str.replace(",", "")

    path = pathlib.Path(path_file).parent
    os.chdir(path)
    bach = open(out_file_name, "w")
    bach.write(batch_str)
    bach.close()
    print("The batch file '{}' is stored in '{}'".format(out_file_name, os.getcwd()))


class CLICommand:
    """
    根据路径文件，创建循环命令， 查看帮助使用 -h。
    可以配合 mgetool findpath 使用，先 findpath 查找路径，再 makebatch 创建循环脚本。
    复杂sh脚本也可以使用该脚本创建，再手动修改具体操作命令。

    如果在 mgetool 中运行:

    Example:

        $ mgetool makebatch -f paths.temp -cmd 'cd $i \necho i \ncd ..'

    如果复制该脚本到某处，仅运行单个脚本:

    Example:

        $ python makebatch.py -f paths.temp -cmd 'cd $i \necho i \ncd ..'

    若希望直接输入路径，请使用 -p 而不是 -f

        $ mgetool makebatch -p “/home/path1 /home/path2” -cmd 'cd $i \necho i \ncd ..'
        $ python makebatch.py -p “/home/path1 /home/path2” -cmd 'cd $i \necho i \ncd ..'


    cmd 命令用单引号。换行使用\n,并且其后不要留有空格。
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--path_file', help='source path file', type=str, default="paths.temp")
        parser.add_argument('-p', '--paths', help='source paths', type=str, default=None)
        parser.add_argument('-cmd', '--command', help='command', type=str, default="cd $i \necho i \ncd ..")
        parser.add_argument('-o', '--store_name', help='out file name', type=str, default="batch.sh")

    @staticmethod
    def run(args, parser):
        make_batch_from_file(args.path_file, args.command, args.store_name)


if __name__ == '__main__':
    # 命令行模式
    import argparse

    parser = argparse.ArgumentParser(description="根据路径文件，创建循环命令："
                                                 "python makebatch.py -cmd 'cd $i \necho i \ncd ..'  "
                                                 "cmd 命令用单引号。换行使用\n,并且其后不要留有空格。")
    parser.add_argument('-f', '--path_file', help='source path file', type=str, default="paths.temp")
    parser.add_argument('-p', '--paths', help='source paths', type=str, default=None)
    parser.add_argument('-cmd', '--command', help='command', type=str, default="cd $i \necho i \ncd ..")
    parser.add_argument('-o', '--store_name', help='out file name', type=str, default="batch.sh")
    args = parser.parse_args()
    if args.paths is None:
        make_batch_from_file(args.path_file, args.command, args.store_name)
    else:
        make_batch(args.paths, args.command, args.store_name)
