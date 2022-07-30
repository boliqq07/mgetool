# -*- coding: utf-8 -*-

# @Time  : 2022/7/30 16:55
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

"""
Notes:
    Import data simply.
"""

import glob
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from warnings import warn

import joblib
import pandas as pd


def _read_txt(file):
    with open(file, mode="r") as f:
        wds = f.readlines()
    return wds


class Call(object):
    """
    Call files in paths, or call files in different paths.

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
            txt=('txt', _read_txt),
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
            list of path.
        backend:str
            default imported type to show.
        prefix_with_upper:bool
            prefix_with_upper for all file add to file in this code to escape same name.
        index_col: int
            use the first column as index in table.
        """

        self._backend = backend
        self._index_col = index_col
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

    def __repr__(self):
        cont_ls = ['<{}> includes:'.format(self.__class__.__name__)]

        for k, v in self._files.items():
            cont_ls.append('"{}": {}'.format(k, v))

        return '\n'.join(cont_ls)

    def __call__(self, *args, **kwargs):
        return self.__extension__[self._backend][1](*args, **kwargs)

    def __getattr__(self, name):
        """
        Returns sub-dataset.

        Parameters
        ----------
        name: str
            Dataset x_name.
        """
        if name in self.__extension__:
            return self.__class__(*self._paths, backend=name, prefix_with_upper=self._prefix)
        else:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

    # This following are used for change backend.
    def csv(self):
        return self.__class__(*self._paths, backend='csv', prefix_with_upper=self._prefix, index_col=self._index_col)

    def pickle_pd(self):
        return self.__class__(*self._paths, backend='pkl_pd', prefix_with_upper=self._prefix, index_col=self._index_col)

    def pickle_sk(self):
        return self.__class__(*self._paths, backend='pkl_sk', prefix_with_upper=self._prefix, index_col=self._index_col)

    def xlsx(self):
        return self.__class__(*self._paths, backend='xlsx', prefix_with_upper=self._prefix, index_col=self._index_col)

    def png(self):
        return self.__class__(*self._paths, backend='png', prefix_with_upper=self._prefix, index_col=self._index_col)

    def jpg(self):
        return self.__class__(*self._paths, backend='jpg', prefix_with_upper=self._prefix, index_col=self._index_col)