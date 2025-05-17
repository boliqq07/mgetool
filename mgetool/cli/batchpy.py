# -*- coding: utf-8 -*-
import os
import pathlib
import shutil

from pathlib import Path


# @Time  : 2023/2/20 3:13
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

_dos_help = r"""在程序py文件中, 对指定的函数名称, 创建遍历指定的路径的batch函数, 进行批量文件的操作。

    batchpy.py -f <file> -n <function_name> -o <output_file_name> -pn <path_name>
    
    -f --file <file> : python file, 需要添加batch函数的python文件
    -n --func_name <function_name> : <file>文件中的函数名称, 针对该函数添加批量处理的函数
        -pn --path_name <path_name> : path name, <function_name> 函数中路径参数的名称, 默认是 path
    -o --out_name <output_file_name> : output file name, 输出的python文件名称, 如果不指定, 则直接在源文件后添加内容，否则创建新文件

    功能逻辑如下：
    
    for i in paths:
        {func_name}(*args, {pn}=i, **kwargs)
    
"""

func_text = r"""

def batch_{func_name}(*args, paths_file="paths.temp", paths=None, batch_verbose=True, **kwargs):
    import pathlib

    if paths is None:
        with open(paths_file) as f:
            w = f.readlines()
        paths = [pathlib.Path(i) for i in w]
    else:
        paths = [pathlib.Path(i) for i in paths]
    
    res = []
    for pi in paths:
        resi = {func_name}(*args, {pn}=pi, **kwargs)
        if batch_verbose:
            print("Done: ", pi)
        res.append(resi)
    return res
    
def batch_{func_name}_pool(*args, paths_file="paths.temp", paths=None, n_jobs=4, mul_method="pool", **kwargs):
    import pathlib

    if paths is None:
        with open(paths_file) as f:
            w = f.readlines()
        paths = [pathlib.Path(i) for i in w]
    else:
        paths = [pathlib.Path(i) for i in paths]
    

    res = []
    
    if mul_method == "pool":
        import multiprocessing
        pool = multiprocessing.Pool(processes=n_jobs)
    
        for pi in paths:
            kwargs["{pn}"] = pi
            resi = pool.apply_async({func_name}, args, kwargs)
            res.append(resi)
        pool.close()
        pool.join()
        res = [i.get() for i in res]
    
    elif mul_method == "joblib":

        from tqdm import tqdm
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed({func_name})(*args, {pn}=pi, **kwargs) for pi in tqdm(paths))
    
    else:
        raise ValueError("mul_method must be pool or joblib")

    
    return res
    

if __name__ == '__main__':
   batch_{func_name}(paths_file=r'paths.temp', paths=None)
   batch_{func_name}_pool(paths_file=r'paths.temp', paths=None, n_jobs=4, mul_method="pool")
        
"""

def batch_py(py_file, func_name, out_name=None,pn="path"):

    
    if out_name is None:
        out_name = py_file
        new = False
    if out_name == py_file:
        new = False
    else:
        new = True
        
    if not os.path.exists(py_file):
        raise FileNotFoundError("file not exist: {}".format(py_file))

    if new:
        func_text2 = func_text.format(func_name=func_name, pn=pn)
        with open(out_name, "w") as f:
            f.write("# -*- coding: utf-8 -*-\n")
            f.write("import sys\n")
            f.write(f'sys.path.append(r"{pathlib.Path(py_file).absolute().parent}")\n')
            f.write("\n")
            f.write("import pathlib\n")
            f.write(f"from {pathlib.Path(py_file).stem} import {func_name}\n")
            f.write(func_text2)
    
    else:
        with open(py_file) as f:
            w = f.readlines()
        func_text2 = func_text.format(func_name=func_name, pn=pn)
        new_twxt = ""
        for i in w:
            new_twxt += i
        new_twxt += "\n"
        new_twxt+= func_text2
        
        with open(py_file, "w") as f:
            f.write(new_twxt)
        


class CLICommand:
    __doc__ = _dos_help

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(dest='file',  help='python file', type=str)
        parser.add_argument('-n', '--func_name', help='function name', type=str)
        parser.add_argument('-o', '--out_name', help='output file name', type=str, default=None)
        parser.add_argument('-pn', '--path_name', help='path name', type=str)
        
        


    @staticmethod
    def run(args, parser):
        if args.file is None:
            parser.print_help()
            return

        if args.func_name is None:
            parser.print_help()
            return
        
        if args.path_name is None:  
            args.path_name = "path"

        batch_py(args.file, args.func_name, out_name=args.out_name, pn=args.path_name)
        print("done")
        

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