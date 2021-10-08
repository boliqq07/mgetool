# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 15:23
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os


def make_batch(paths, cmd="cd $i \n echo i \n cd .."):
    batch_str = """#!/bin/bash
    echo $dirname
    
    old_path = $dirname
    
    for i in {}
    
    {}
    
    cd old_path
    
    done
        """.format(paths, cmd)

    batch_str = batch_str.replace("'", "")
    batch_str = batch_str.replace("[", "")
    batch_str = batch_str.replace("]", "")
    batch_str = batch_str.replace(",", "")
    bach = open("batch.sh", "w")
    bach.write(batch_str)
    bach.close()
    print("The batch file is stored in {}".format(os.getcwd()))


def make_batch_from_file(path_file, cmd="cd $i \n echo i \n cd .."):
    batch_str = """#!/bin/bash
    echo $dirname
    
    old_path = $dirname
    
    for i in {}
    
    {}
    
    cd old_path
    
    done
        """.format(path_file, cmd)

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

        $ mgetool makebatch -f paths.temp -cmd cd $i qsub run.lsfpbs
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--path_file', help='source path file',type=str, default=None)
        parser.add_argument('-cmd', '--command', help='command',type=str,default="cd $i \n echo i \n cd ..")

    @staticmethod
    def run(parser):
        parser = parser.parse_args()
        make_batch(parser.path_file, parser.command)