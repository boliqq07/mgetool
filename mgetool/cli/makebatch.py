# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 15:23
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os
import pathlib

_dos_help = r"""根据路径文件,创建循环命令, 查看帮助使用 -h.
可以配合 mgetool findpath 使用,先 findpath 查找路径,再 makebatch 创建循环脚本.
复杂sh脚本也可以使用该脚本创建,再手动修改具体操作命令.

运行方式1:

    $ makebatch
    
    $ makebatch 'jsub < $(find -name *.run)'
    
运行方式2: 如果复制此文件到某处,仅运行单个脚本:

    $ python makebatch.py -f paths2.temp
    
    $ python makebatch.py 'cd $i \necho i \ncd ..'

Key 1:

    -cmd 后的命令用单引号包裹,换行使用\n,并且其后不要留有空格,默认单引号包裹, -cmd 本身可以省略。
    （若不包裹,字符首先由系统shell解释,再传递到python, 请确定您知悉自己的操作目的再作决定.）

Key 2:

    -r 将直接运行循环命令，请有一定的自信再操作。

例：

    # 批量复制
    $ makebatch 'cp '$PWD'/cpu.run ../ini_opt' -o cpbatch.sh

    # 提交脚本
    $ makebatch -cmd 'jsub < $(find -name *.run)'

    # vasp结构优化结果转vasp静态计算输入
    $ makebatch 'cd .. \nrm -rf ini_static \ncp -r ini_opt ini_static \ncp ini_opt/CONTCAR ini_static/POSCAR \ncp '$PWD'/static_INCAR ini_static/INCAR' -o static_ini.sh

    # vasp结构优化结果转vasp静态计算输入
    $ makebatch 'cd .. \nrm -rf pure_static \ncp -r pure_opt pure_static \ncp pure_opt/CONTCAR pure_static/POSCAR \ncp '$PWD'/static_INCAR pure_static/INCAR' -o static_pure.sh

    # neb过渡态计算生成
    $ makebatch 'cd .. \nnebmake.pl ini_static/CONTCAR fin_static/CONTCAR 3 \ncp ini_static/OUTCAR 00/OUTCAR \ncp fin_static/OUTCAR 04/OUTCAR \ncp ini_static/KPOINTS KPOINTS\ncp ini_static/POTCAR POTCAR \ncp '$PWD'/neb_cpu.run neb_cpu.run' -o nebbatch.sh
"""


def trans_str(string):
    lines = string.split("\\n")
    [print(i) for i in lines]
    lines = "\n".join(lines)
    return lines


def make_batch_from_file(path_file, cmd_arg="", cmd="", out_file_name="batch.sh", keep_path=False, run_sh=False):
    if cmd_arg != "":
        cmd = cmd_arg

    if keep_path:
        batch_str = """#!/bin/bash

for i in $(cat {})

do
echo $i

{}

done
        """.format(path_file, trans_str(cmd))
    else:
        batch_str = """#!/bin/bash

old_path=$(cd "$(dirname "$0")"; pwd)

for i in $(cat {})

do
cd $i
echo $(cd "$(dirname "$0")"; pwd)

{}

cd $old_path
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
    print("The batch file '{}' is stored in '{}' .".format(out_file_name, os.getcwd()))
    if run_sh:
        try:
            print(f"Run batch file '{out_file_name}':")
            os.system(f"sh {out_file_name}")
        except Exception as e:
            print(f"Try to run batch file '{out_file_name}', but failed, check the script manually.")
            raise e


class CLICommand:
    __doc__ = _dos_help

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(dest='cmd_arg', help='commands.', nargs="?", type=str, default="")
        parser.add_argument('-cmd', '--commands', help='commands.', type=str, default="")
        parser.add_argument('-f', '--path_file', help='source path file.', type=str, default="paths.temp")
        parser.add_argument('-k', '--keep_path', help='not enter the sub-dirs.', action="store_true")
        parser.add_argument('-o', '--store_name', help='out file name.', type=str, default="batch.sh")
        parser.add_argument('-r', '--run_sh', help='run the batch directly.', action="store_true")

    @staticmethod
    def run(args, parser):
        make_batch_from_file(args.path_file, args.cmd_arg, args.commands, args.store_name, args.keep_path, args.run_sh)

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()


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
