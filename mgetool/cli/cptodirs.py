# -*- coding: utf-8 -*-
import os
import shutil

from pathlib import Path


# @Time  : 2023/2/20 3:13
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

_dos_help = r"""将固定文件，循环复制到多个目录中。

1.
    cptodirs _formatter.py -pf paths.temp -c _formatter.py:for.py -c asfa:fas -v
    
注意：路径文件请不要留有空行。

"""

def copyfile(files, w, name_pair=None,verbose=False):
    
    if name_pair is None:
        name_pair = {Path(i).name: Path(i).name for i in files}
    
    for file in files:
        des_name = name_pair.get(file.name, file.name)
        for dest_dir in w:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / des_name
            shutil.copy2(file, dest_file)
            if verbose:
                print(f"Copied {file} to {dest_file}")
                
def delfile(files, w, name_pair=None, verbose=False):
    if name_pair is None:
        name_pair = {Path(i).name: Path(i).name for i in files}
    
    for file in files:
        des_name = name_pair.get(file.name, file.name)
        for dest_dir in w:
            dest_file = dest_dir / des_name
            if os.path.exists(dest_file):
                os.remove(dest_file)
                if verbose:
                    print(f"Deleted {dest_file}")
            else:
                if verbose:
                    print(f"{dest_file} does not exist.")
            


class CLICommand:
    __doc__ = _dos_help

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(dest='file', nargs="+",  help='file or file list', type=str)
        parser.add_argument('-pf', '--path_file', help='files contain paths.', type=str, default="paths.temp")
        parser.add_argument('-d', '--delete', help='del the copyed file', action="store_true")
        parser.add_argument('-c', '--change_name_pair', help='name pair or paris', nargs="+",  type=str, default=None)
        parser.add_argument('-v', '--verbose', help='verbose mode', action="store_true")
        


    @staticmethod
    def run(args, parser):

        with open(args.path_file) as f:
            w = f.readlines()
        if len(w) == 0:
            return
        w = [i.rstrip() for i in w]
        w = [Path(i) for i in w]
        
        if isinstance(args.file, str):
            files = [args.file]
        else:
            files = args.file
        files  = [Path(i) for i in files]
        
        if args.change_name_pair is not None:
            name_pair = [args.change_name_pair] if isinstance(args.change_name_pair, str) else args.change_name_pair
            
            name_pair = {i.split(":")[0]: i.split(":")[1] for i in name_pair}
        else:
            name_pair = None
        
        if args.delete:
            delfile(files, w, name_pair, verbose=args.verbose)
        else:
            copyfile(files, w, name_pair, verbose=args.verbose)

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()


def main():
    """

    """

    from mgetool.cli._formatter import Formatter
    import argparse

    parser = argparse.ArgumentParser(description=_dos_help, formatter_class=Formatter)
    CLICommand.add_arguments(parser=parser)
    args = CLICommand.parse_args(parser=parser)
    CLICommand.run(args=args, parser=parser)


if __name__ == '__main__':
    main()