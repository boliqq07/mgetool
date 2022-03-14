#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/29 19:47
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause


"""
Notes:
    import data simply.
    # Just a copy from xenonpy
    for pictures use skimage.io.ImageCollection

"""
import copy
import glob
import os
import re
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path
from warnings import warn

import joblib
import pandas as pd
import requests
from tqdm import tqdm

from mgetool.tool import def_pwd, parallelize


class Call(object):
    """
    Call file in paths.

    When there are four files in pwd path:
        (file1.csv, file2.csv, file3.txt, file4.png)

    Examples
    ---------
    >>> call = Call(".",backend="csv")
    >>> file1 = call.file1
    >>> file2 = call.file2
    >>> call = Call(".",backend="txt")
    >>> file = call.file3

    """

    @staticmethod
    def extension(index_col=0):

        read_csv = partial(pd.read_csv, index_col=index_col)
        read_excel = partial(pd.read_excel, index_col=index_col)
        extension = dict(
            pkl_pd=('pkl_pd', pd.read_pickle),
            csv=('csv', read_csv),
            xlsx=('xlsx', read_excel),
            pkl_sk=('pkl_sk', joblib.load),
        )
        try:
            from skimage import io
            extension.update({"png": ("png", io.imread), "jpg": ("jpg", io.imread)})
        except ImportError:
            pass
        return extension

    __re__ = re.compile(r'[\s\-.]')

    def __init__(self, *paths, backend='pkl_pd', prefix_with_upper=None, index_col=0):
        """

        Parameters
        ----------
        paths:str
            list of path
        backend:str
            default imported type to show
        prefix_with_upper:str
            prefix_with_upper for all file add to file in this code to escape same name
        index_col:
            use the first column as index in table.
        """

        self._backend = backend
        self.index_col = index_col
        self._files = None
        self.__extension__ = self.extension(index_col)

        if len(paths) == 0:
            self._paths = ('.',)
        else:
            self._paths = paths

        if not prefix_with_upper:
            prefix_with_upper = ()
        self._prefix = prefix_with_upper

        self._make_index(prefix_with_upper=prefix_with_upper)

    def _make_index(self, *, prefix_with_upper):
        def make(path_):
            patten = self.__extension__[self._backend][0]
            files = glob.glob(str(path_ / ('*.' + patten)))

            def _nest(_f):
                f_ = _f
                return lambda s: s.__extension__[s._backend][1](f_)

            for f in files:
                # selection data_cluster
                f = Path(f).resolve()
                parent = re.split(r'[\\/]', str(f.parent))[-1]
                # parent = str(f.parent).split('\\/')[-1]
                fn = f.name[:-(1 + len(patten))]
                fn = self.__re__.sub('_', fn)
                if prefix_with_upper:
                    fn = '_'.join([parent, fn])

                if fn in self._files:
                    warn("file %s with x_name %s already bind to %s and will be ignored" %
                         (str(f), fn, self._files[fn]), RuntimeWarning)
                else:
                    self._files[fn] = str(f)
                    setattr(self.__class__, fn, property(_nest(str(f))))

        self._files = defaultdict(str)
        for path in self._paths:
            path = Path(path).expanduser().absolute()
            if not path.exists():
                raise RuntimeError('%s not exists' % str(path))
            make(path)

    @classmethod
    def from_http(cls, url, save_to, *, filename=None, chunk_size=256 * 1024, params=None,
                  **kwargs):
        """
        Get file object via a http request.

        Parameters
        ----------
        url: str
            The resource url.
        save_to: str
            The path of a path to save the downloaded object into it.
        filename: str, optional
            Specific the file x_name when saving.
            Set to ``None`` (default) to use a inferred x_name from http header.
        chunk_size: int, optional
            Chunk size.
        params: any, optional
            Parameters will be passed to ``requests.get`` function.
            See Also: `requests <http://docs.python-requests.org/>`_
        kwargs: dict, optional
            Pass to ``requests.get`` function as the ``kwargs`` parameters.

        Returns
        -------
        str
            File path contains file x_name.
        """
        r = requests.get(url, params, **kwargs)
        r.raise_for_status()

        if not filename:
            if 'filename' in r.headers:
                filename = r.headers['filename']
            else:
                filename = url.split('/')[-1]

        if isinstance(save_to, str):
            save_to = Path(save_to)
        if not isinstance(save_to, Path) or not save_to.is_dir():
            raise RuntimeError('%s is not a legal path or not point to a path' % save_to)

        file_ = str(save_to / filename)
        with open(file_, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        return file_

    def __repr__(self):
        cont_ls = ['<{}> includes:'.format(self.__class__.__name__)]

        for k, v in self._files.items():
            cont_ls.append('"{}": {}'.format(k, v))

        return '\n'.join(cont_ls)

    def csv(self):
        return Call(*self._paths, backend='csv', prefix_with_upper=self._prefix, index_col=self.index_col)

    def pickle_pd(self):
        return Call(*self._paths, backend='pkl_pd', prefix_with_upper=self._prefix, index_col=self.index_col)

    def pickle_sk(self):
        return Call(*self._paths, backend='pkl_sk', prefix_with_upper=self._prefix, index_col=self.index_col)

    def xlsx(self):
        return Call(*self._paths, backend='xlsx', prefix_with_upper=self._prefix, index_col=self.index_col)

    def png(self):
        return Call(*self._paths, backend='png', prefix_with_upper=self._prefix, index_col=self.index_col)

    def jpg(self):
        return Call(*self._paths, backend='jpg', prefix_with_upper=self._prefix, index_col=self.index_col)

    def __call__(self, *args, **kwargs):
        return self.__extension__[self._backend][1](*args, **kwargs)

    def __getattr__(self, name):
        """
        Returns sub-dataset.

        Parameters
        ----------
        name: str
            Dataset x_name.

        Returns
        -------
        spath
        """
        if name in self.__extension__:
            return self.__class__(*self._paths, backend=name, prefix_with_upper=self._prefix)
        else:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))


def check_file(spath, file_path, suffix=None):
    os.chdir(file_path)
    # print(os.path.abspath(os.curdir))
    all_file = os.listdir('.')
    files = []
    for f in all_file:
        if os.path.isdir(f):
            ff = os.path.join(file_path, f)
            files.extend(check_file(spath, ff, suffix=suffix))
            os.chdir(file_path)
        else:
            if not suffix:
                che = True
            elif suffix == "":
                che = "" == os.path.splitext(f)[1]
            else:
                che = "".join((".", suffix)) == os.path.splitext(f)[1]
            if che:
                rel_path = file_path.replace(spath, "")
                parents = re.split(r'[\\/]', str(rel_path))
                files.append([parents, f])
            else:
                pass
    return files


class BatchFile:
    r"""
    Search files and filter files and re-site files.

    Examples
    ---------
    >>> a = BatchFile(".")
    >>> a.filter_dir_name("a")
    >>> a.filter_file_name("2")
    >>> print(a.file_list)
    ...
    #copy the file to new path and keep the dir structure
    >>> a.to_path(r"C:\Users\Admin\Desktop\d2",flatten=False)
    #copy the file to new path, flatten the file and add the dir name on file: dirname_1_filename.
    >>> a.to_path(r"C:\Users\Admin\Desktop\d2", add_dir=[-1], flatten=True)
    #copy the file to new path, flatten the file and add the dir name on file: dirname_2_dirname_1_filename.
    >>> a.to_path(r"C:\Users\Admin\Desktop\d2", add_dir=[-2,-1], flatten=True)

    """

    def __init__(self, path=None, suffix=None, fdir_range="all"):
        """

        Parameters
        ----------
        path:str
            total dir of all file
        suffix:str
            suffix of file
            Examples:
                .txt
        fdir_range:str,tuple,list
            dir range to find.
        """

        path = def_pwd(path, change=True)
        self.path = path

        parents = re.split(r'[\\/]', str(path))
        self.parents = parents

        fm = {
            "all": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            "now": (1,),
            "next": (2,),
            "next2": (3,),
            "next3": (4,),
            "to-next": (1, 2),
            "to-next2": (1, 2, 3),
            "to-next3": (1, 2, 3, 4),
        }

        if isinstance(fdir_range, str):
            if fdir_range in fm:
                fdir_range = fm[fdir_range]
            else:
                raise NotImplementedError("Can't accept {}".format(fdir_range))

        else:
            assert isinstance(fdir_range, (list, tuple)), "please set which layer of dir to find."

        self.fdir_range = fdir_range

        self.file_list_old = check_file(path, path, suffix=suffix)
        self.file_list = [i for i in self.file_list_old if len(i[0]) in fdir_range]
        self.init_file = tuple(self.file_list)
        self.file_list_merge = []
        self.file_list_merge_new = []
        self.file_dir = []

    def filter_file_name(self, include=None, exclude=None):
        """

        Parameters
        ----------
        include:str,list
            get the filename with include str
            such as hold "ast_tep" with "ast" string
        exclude: str,list
            delete the filename with exclude str
            such as hold "ast_cap" and delete "ast_tep" with "tep" str,

        """

        if include is None and exclude is None:
            return

        assert include != []
        assert exclude != []
        if isinstance(include, str):
            include = [include, ]
        if isinstance(exclude, str):
            exclude = [exclude, ]

        file_list_filter = []
        for file_i in self.file_list:
            name = file_i[1]

            if include and not exclude:
                if any([i in name for i in include]):
                    file_list_filter.append(file_i)
            elif not include and exclude:
                if not any([i in name for i in exclude]):
                    file_list_filter.append(file_i)
            elif include and exclude:
                if any([i in name for i in include]) and not any([i in name for i in exclude]):
                    file_list_filter.append(file_i)
            else:
                raise TypeError("one of include, exclude must be str or list of str")

        self.file_list = file_list_filter

    def filter_dir_name(self, include=None, exclude=None, layer=-1):
        """
        Filter the dir(and its sub_file).

        Parameters
        ----------
        include:str,list of str
            get the filename with include str
            such as hold "ast_tep" with "ast" string
        exclude: str, list of str
            delete the filename with exclude str
            such as hold "ast_cap" and delete "ast_tep" with "tep" str,
        layer:int,list
            if list, check the sum name of the layers.
            Filter dir with target layer,all the dir should contain the sublayer!
            Examples:
                for /home,
                /home/ast, -1
                /home/ast/eag, -2
                /home/ast/eag/kgg, -3

        """
        if include is None and exclude is None:
            return

        assert include != []
        assert exclude != []
        if isinstance(include, str):
            include = [include, ]
        if isinstance(exclude, str):
            exclude = [exclude, ]

        file_list_filter = []

        for file_i in self.file_list:
            try:
                if isinstance(layer, int):
                    layer = [layer, ]

                if isinstance(layer, list):
                    name = [file_i[0][i] for i in layer]
                else:
                    name = file_i[0]
                name = "".join(name)

                if include and not exclude:
                    if any([i in name for i in include]):
                        file_list_filter.append(file_i)
                elif not include and exclude:
                    if not any([i in name for i in exclude]):
                        file_list_filter.append(file_i)
                elif include and exclude:
                    if any([i in name for i in include]) and not any([i in name for i in exclude]):
                        file_list_filter.append(file_i)
                else:
                    raise TypeError("one of include, exclude must be str or list of str")
            except IndexError:
                pass

        self.file_list = file_list_filter

    def merge(self, path=None, flatten=False, add_dir="3-layer", refresh_file_list=True, pop=0):
        """
        Merge dir and file name together.

        Parameters
        ----------
        path:str
            new path
        flatten:True
            flatten the filtered file.
            if flatten is dict, the key is the specific dir name,and value is True.
            Examples:
            flatten = {"asp":True}
        add_dir:int,list
            add the top dir_name to file to escape same name file.
            only valid for flatten=True
        refresh_file_list:bool
            refresh file_list or not.
        pop: int (negative)
            pop the last n layer. default =0
            used for copy by dir rather than files. just used for flatten=False

        Returns
        -------
            new filename

            Args:
                refresh_file_list:
                refresh_file_list:
        """
        if not path:
            path = self.path
            flatten = False
        if not add_dir:
            add_dir = []
        elif add_dir == "3-layer":
            add_dir = [-1, -2, -3]
        if isinstance(add_dir, int):
            add_dir = [add_dir, ]

        if flatten is not False:
            assert pop == 0
        assert pop <= 0

        file_list_merge = []
        for file_i in self.file_list:
            site = copy.copy(file_i[0])
            if isinstance(flatten, dict):
                site = [site[_] for _ in add_dir]
                site_c = ""
                for i, j in enumerate(site):
                    i -= len(site)
                    if i in flatten.keys():
                        if flatten[i] in [True, "layer", "dir", "folder", 1, "s"]:
                            site_c += "".join((j, "/"))
                        else:
                            site_c += "".join((j, "_"))
                    else:
                        site_c += "".join((j, "_"))
                site_c = re.split(r'[\\/]', str(site_c))
                site_c[-1] += file_i[1]
                file_list_merge.append(os.path.join(path, *site_c))

            elif flatten:
                site = [site[_] for _ in add_dir]
                site.append(file_i[1])
                site = "_".join(site)
                file_list_merge.append(os.path.join(path, site))
            else:

                site.append(file_i[1])
                if pop != 0:
                    site = site[:pop]
                namei = os.path.join(path, *site)

                if len(file_list_merge) == 0 or namei != file_list_merge[-1]:
                    file_list_merge.append(namei)

        if refresh_file_list:
            self.file_list_merge = file_list_merge
        fdir = list(set([os.path.dirname(i) for i in file_list_merge]))
        fdir.sort()
        self.file_dir = fdir
        return file_list_merge

    def to_path(self, new_path, flatten=False, add_dir="3-layer", pop=0, n_jobs=1):
        """

        Parameters
        ----------
        new_path:str
            new path
        flatten:bool,dict
            flatten the filtered file.
            if flatten is dict, the key is the specific dir name,and value is True.
            Examples:
            flatten = {"asp":True}
        add_dir:list, int
            add the top dir_name to file to escape same name file.
            only valid for flatten=True
        pop: int (negative)
            pop the last n layer. default =0
            used for copy by dir rather than files. just used for flatten=False
        n_jobs:int
            n_jobs

        Returns
        -------
            file in path.
        """
        self.file_list_merge = self.merge(pop=pop)
        new_path = def_pwd(new_path)
        self.file_list_merge_new = self.merge(path=new_path, flatten=flatten, add_dir=add_dir,
                                              refresh_file_list=False, pop=pop)
        if len(set(self.file_list_merge_new)) < len(set(self.file_list_merge)):
            raise UserWarning("There are same name files after flatten folders. "
                              "you can change add_dir to add difference prefix to files", )
        if n_jobs != 1:
            parallelize(n_jobs, self.copy_user, zip(self.file_list_merge, self.file_list_merge_new, ),
                        mode="j",
                        respective=False)
        else:
            for ij in tqdm(list(zip(self.file_list_merge, self.file_list_merge_new))):
                self.copy_user(ij)

    @staticmethod
    def copy_user(k):
        i, j = k
        if os.path.isdir(i):
            shutil.copytree(i, j)
        else:
            path_i = os.path.split(j)[0]
            if not os.path.exists(path_i):
                os.makedirs(path_i)
            shutil.copy(i, j)

# if __name__ == "__main__":
# others please use shutil
# shutil.copytree()
# a = BatchFile(r"C:\Users\wangchangxin\Desktop\d1")
# a.filter_dir_name("a", layer=-1)
# a.filter_file_name("2")
# a.to_path(r"C:\Users\wangchangxin\Desktop\d2", add_dir=[-2, -1], flatten=True)
# bf = BatchFile(r"/home/iap13/wcx/CHG")

# bf.filter_dir_name(include="Mo")
# filenames = bf.file_list
