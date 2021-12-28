# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 15:23
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os


def trans_str(string):
    lines = string.split("\\n")
    print(lines)
    lines = "\n".join(lines)
    return lines



def make_batch(paths, cmd="cd $i \necho i \ncd .."):
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
    bach = open("batch.sh", "w")
    bach.write(batch_str)
    bach.close()
    print("The batch file is stored in {}".format(os.getcwd()))


def make_batch_from_file(path_file, cmd="cd $i \necho i \ncd .."):
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
    bach = open("batch.sh", "w")
    bach.write(batch_str)
    bach.close()
    print("The batch file is stored in {}".format(os.getcwd()))


class CLICommand:

    """
    根据路径文件，创建循环命令。


    Example:

        $ mgetool makebatch -f paths.temp -cmd 'cd $i \necho i \ncd ..'

    cmd 命令用单引号。换行使用\n,并且其后不要留有空格
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--path_file', help='source path file',type=str, default=None)
        parser.add_argument('-cmd', '--command', help='command',type=str, default="cd $i \necho i \ncd ..")

    @staticmethod
    def run(args, parser):
        args = parser
        make_batch_from_file(args.path_file, args.command)


if __name__ == '__main__':
    # 命令行模式
    import argparse

    parser = argparse.ArgumentParser(description="根据路径文件，创建循环命令："
                                                 "mgetool makebatch -f paths.temp -cmd 'cd $i \necho i \ncd ..'  "
                                                 "cmd 命令用单引号。换行使用\n,并且其后不要留有空格")
    parser.add_argument('-f', '--path_file', help='source path file', type=str, default=None)
    parser.add_argument('-cmd', '--command', help='command', type=str, default="cd $i \necho i \ncd ..")
    args = parser.parse_args()
    make_batch_from_file(args.path_file, args.command)