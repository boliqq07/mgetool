#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/29 19:48
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause


"""
Notes:
    export data simply
    # Just a copy from xenonpy


"""

import os
import sys
from os import remove

import joblib
import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm


def def_pwd(path):
    if path is None:
        path = os.getcwd()

    if os.path.exists(path):
        os.chdir(path)
    else:
        os.makedirs(path)
        os.chdir(path)
    pwd = os.getcwd()
    locals()[pwd] = pwd
    return pwd


class Store(object):
    """
    Store file to path.

    Default mode is "w" ,which can be"a+" in txt.
    'w'       create a new file, open for writing, clear contents if it exists.
    'a+'      open for writing, appending to the end of the file if it exists.
    'n'       create a new file and open it for writing, the name are set by number.
    """

    def __init__(self, path=None, filename="filename", prefix: str = None):
        """

        Parameters
        ----------
        path:str
            /data_cluster, or F:data_cluster/data1
        filename:str
            filename
        prefix:str
            prefix for all filenname

        """

        if not prefix:
            prefix = ""
        self._prefix = prefix

        def_pwd(path)

        self._path = path
        self._filename = ""
        self.default_filename = filename
        self._file_list = []

    def __repr__(self):
        return "store to ({}) with {} file".format(self._path, len(self.stored_file))

    __str__ = __repr__

    def _check_name(self, suffix="csv", file_new_name="filename", mode="w"):

        self._filename = file_new_name or self.default_filename

        if os.path.isfile('{}{}.{}'.format(self._prefix, self._filename, suffix)) and mode == "n":
            shu1 = 1
            while os.path.isfile('{}{}({}).{}'.format(self._prefix, self._filename, shu1, suffix)):
                shu1 += 1
            self._filename = '{}{}({}).{}'.format(self._prefix, self._filename, shu1, suffix)
        else:
            self._filename = '{}{}.{}'.format(self._prefix, self._filename, suffix)

        if self._filename in self._file_list:
            self._file_list.remove(self._filename)
        self._file_list.append(self._filename)

    def to_csv(self, data, file_new_name=None, mode="w", transposition=False):
        """
        Parameters
        ----------
        data: object
            data.
        file_new_name:str
            file name, if None, default is "filename(i)".
        mode: str
            "w" or "a+"
        transposition:
             transposition the table

        """
        self._check_name("csv", file_new_name, mode=mode)
        if mode == "n":
            mode = "w"
        if isinstance(data, (dict, list)):
            data = pd.DataFrame.from_dict(data).T

        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        if isinstance(data, pd.DataFrame):
            if transposition:
                data = data.T
            data.to_csv(path_or_buf="%s" % self._filename, sep=",", na_rep='', float_format=None,
                        columns=None, header=True, index=True, index_label=None,
                        mode=mode, encoding=None, )

        else:
            raise TypeError("Not support data_cluster type:%s for csv" % type(data))

    def to_txt(self, data, file_new_name=None, mode="w"):
        """
        Parameters
        ----------
        data: object
            data.
        file_new_name:str
            file name, if None, default is "filename(i)".
        mode: str
            "w" or "a+" or "n"

        """
        self._check_name("txt", file_new_name, mode=mode)
        if mode == "n":
            mode = "w"
        document = open(self._filename, mode=mode)
        document.write(str(data))
        document.close()

    def to_pkl_pd(self, data, file_new_name=None, mode="n"):
        """

        Parameters
        ----------
        data: object
            data.
        mode: str
            ‘n’
        file_new_name:str
            file name, if None, default is "filename(i)".

        """
        self._check_name("pkl_pd", file_new_name, mode=mode)
        pd.to_pickle(data, self._filename)

    def to_pkl_sk(self, data, file_new_name=None, mode="n"):
        """

        Parameters
        ----------
        data: object
            data.
        mode: str
            ‘n’
        file_new_name:str
            file name, if None, default is "filename(i)".

        """
        self._check_name("pkl_sk", file_new_name, mode=mode)
        joblib.dump(data, self._filename)

    def to_png(self, data, file_new_name=None):
        """

        Parameters
        ----------
        data:object
            data.
        file_new_name:str
            file name, if None, default is "filename(i)".

        """
        self._check_name("png", file_new_name=file_new_name)
        io.imsave(self._filename, data)

    @classmethod
    def to_multi_file(cls, data_s, suffix="pkl_sk", file_new_name=None, mode="w"):
        """
        Store a series files.

        Parameters
        ----------
        data_s:list of object
            list of data. The single data, should be set in [].
        suffix: str
            file type {"txt": cls.to_txt, "pkl_sk": cls.to_pkl_sk, "pkl_pd": cls.to_pkl_pd, "csv": cls.to_csv,
            "png": cls.to_png}.
        file_new_name:str
            file name, if None, default is "filename(i)".
        mode: str
            "w" , "a+","n".

        """
        dict_func = {"txt": cls.to_txt, "pkl_sk": cls.to_pkl_sk, "pkl_pd": cls.to_pkl_pd,
                     "csv": cls.to_csv, "png": cls.to_png}
        if suffix in ["pkl_sk", "pkl_pd", "png"] and mode == "a+":
            raise UserWarning(
                '"pkl_sk","pkl_pd","png" just accept mode="w" or "n",the file would be stored respectively')

        for data in tqdm(data_s):
            dict_func[suffix](data, file_new_name, mode=mode)

    def remove(self, index_or_name=None):
        """
        Remove the indexed file.

        Parameters
        ----------
        index_or_name:int or str
            index or x_name,default=-1

        """

        if isinstance(index_or_name, str):
            name = index_or_name
            index = -1
        elif isinstance(index_or_name, int):
            name = None
            index = index_or_name
        else:
            name = None
            index = -1
        if not name:
            try:
                files = self._file_list[index]
            except IndexError:
                raise IndexError("No flie or wrong index to remove")
            else:
                if not isinstance(files, list):
                    remove(str(files))
                else:
                    for f in files:
                        remove(str(f))
                del self._file_list[index]

        elif name in self._file_list:
            if not isinstance(name, list):
                remove(str(name))
                self._file_list.remove(name)
            else:
                for f in name:
                    remove(str(f))
                    self._file_list.remove(f)
                self._file_list.remove([])

        else:
            raise NameError("No flie named %s" % index_or_name)

    def withdraw(self):
        """
        Delete all stored_file file.
        """
        files = self._file_list
        for f in files:
            remove(str(f))
        self._file_list = []
        self._filename = ""

    @property
    def stored_file(self):
        """show the stored file"""
        [print(i) for i in self._file_list]
        return self._file_list

    def start(self, file_new_name="print_log", mode="w"):
        """
        Parameters
        ----------
        data: object
            data.
        file_new_name:str
            file name, if None, default is "filename(i)".
        mode: str
            "w" or "a+" or "n"

        """

        self._check_name("txt", file_new_name, mode=mode)

        if mode == "n":
            mode = "w"

        sys.stdout = Logger(self._filename, mode=mode)

    def end(self):
        try:
            sys.stdout.log.close()
        except AttributeError:
            pass


class Logger(object):
    def __init__(self, filename="Default.log", mode="a+"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == "__main__":

    st = Store()
    st.start()

    print(os.path.dirname(__file__))
    print('------------------')
    for i in tqdm(range(5, 10)):
        print("this is the %d times" % i)

    st.end()
