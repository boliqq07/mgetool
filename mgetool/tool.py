#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/29 19:44
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
Notes:
    Some tools for characterization.
"""
import inspect
import itertools
import multiprocessing
import numbers
import os
import random
import time
from collections.abc import Iterable
from functools import partial, wraps
from itertools import chain
from sys import getsizeof
from typing import Union

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from tqdm import tqdm


def time_this_function(func):
    """
    Time the function.
    use as a decorator.

    Examples
    ---------
    ::
        @time_this_function

        def func(x):
            return x
        a= func(1)

    Parameters
    ----------
    func: Callable
        function

    Returns
    -------
    result
        function results
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, "time", end - start)
        return result

    return wrapper


def check_random_state(seed):
    """
    Turn seed into a random.RandomState instance.
    if using the numpy random, please user the check_random_state in sklearn.

    Parameters
    ----------
    seed: None,int or RandomState
        If seed is None, return the RandomState singleton used by random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise, raise ValueError.

    Returns
    -------
    random.Random
        RandomState object
    """

    if seed is None or seed is random.random:
        return random.Random()
    if isinstance(seed, (numbers.Integral, np.integer)):
        return random.Random(seed)
    if isinstance(seed, random.Random):
        return seed
    raise ValueError('%r cannot be used to seed a seed'
                     ' instance' % seed)


def tqdm2(iterable, tq=True, desc=None, **kwargs):
    if tq:
        return tqdm(iterable, desc=desc, **kwargs)
    else:
        return iterable


class VirPool:
    """Just keep same operation."""

    def close(self):
        pass

    def join(self):
        pass


def funcz(args, ff=None, respective=False, respective_kwargs=False, **kwargs):
    """Keep the site of this function."""
    if respective:
        if respective_kwargs:
            *i, kw = args
            return ff(*i, **kw, **kwargs)
        else:
            return ff(*args, **kwargs)
    else:
        return ff(args, **kwargs)


def funcz_batch(iterablei, ff=None, respective=False, respective_kwargs=False, **kwargs):
    if respective:
        if respective_kwargs:
            return [ff(*i, **kw, **kwargs) for *i, kw in list(iterablei)]
        else:
            return [ff(*i, **kwargs) for i in list(iterablei)]
    else:
        return [ff(i, **kwargs) for i in list(iterablei)]


def parallelize(n_jobs, func, iterable, respective=False, tq=True, batch_size: Union[int, str] = 'auto',
                store=None, mode="j", lazy=False,
                parallel_para_dict=None, respective_kwargs=False, desc=None, chunksize=1,
                **kwargs):
    """
    Parallelize the function for iterable.
    Make sure the 'func' youself is before the ' if __name__ == "__main__": ' code.

    Examples
    ----------
    >>> def func(x):
    ...     return x**2
    >>> if __name__=="__main__":
    >>>     result = parallelize(n_jobs=2,func=func,iterable=[1,2,3,4,5])
    [1,2,3,4,5]

    Note:
        For "large" calculated function, with small return for each function.

        def func:
            'large' calculation.
            ...
            ...
            ...
            return int

    make sure the if __name__ == "__main__":

    Parameters
    ----------
    parallel_para_dict:dict
        Parameters passed to joblib.Parallel, mode="j"
    batch_size:str,int
        size for mode="j"
    respective:bool
        Import the parameters respectively or as a whole for each one of iterable object.
    respective_kwargs:
        the respective parameters contains kwargs or not. only for respective=True
        # >>> for iter_i, kw in tqdm(iterable)):
        # >>>    func(*iter_i,**kw)

    tq:bool
         view progress or not
    desc:str
        prefix for the progressbar.
    n_jobs:int
        cpu numbers. n_jobs is the number of workers requested by the callers. Passing n_jobs=-1
    means requesting all available workers for instance matching the number of CPU cores on the worker host(s).
    func:
        function to calculate.
    iterable:
        iterable object.
    kwargs:
        stable kwargs for 'func' function.
    store:
        Not been used.
    mode:
        "j":"joblib" or "m":"multiprocessing"
        all model ["apply", "apply_async", "map", "starmap","starmap_async", "map_async", "imap", "im", "j", "joblib"]
    lazy: bool
        return generator (lazy=True) or result list (lazy=False).

    Returns
    -------
    results
        function results


    """
    if mode == "auto":
        if lazy:
            if respective:
                if respective_kwargs:
                    mode = "apply"
                else:
                    mode = "starmap"
            else:
                mode = "imap"  # or  "map"
        else:
            # mode = "j"
            if respective:
                if respective_kwargs:
                    mode = "apply_async"
                else:
                    mode = "starmap_async"
            else:
                mode = "map_async"

    _ = store  # Not used.

    if respective_kwargs:
        assert respective is True
        if effective_n_jobs(n_jobs) != 1:
            assert mode in ["apply", "apply_async", "j", "joblib", "m", "imap", "im"]

    if mode == "m":
        mode = "imap"  # old version match.

    if effective_n_jobs(n_jobs) == 1:

        if respective:
            if respective_kwargs:
                result = (
                    func(*iter_i, **kws, **kwargs) for *iter_i, kws in tqdm2(iterable, tq=tq, desc=desc))
            else:
                result = (func(*iter_i, **kwargs) for iter_i in tqdm2(iterable, tq=tq, desc=desc))
        else:
            result = (func(iter_i, **kwargs) for iter_i in tqdm2(iterable, tq=tq, desc=desc))

        if lazy:
            return result, VirPool()
        else:
            return [i for i in result]

    if mode in ["j", "joblib"]:

        assert lazy is False, f"Just support lazy=False for mode={mode}."
        if parallel_para_dict is None:
            parallel_para_dict = {}
        para_func = Parallel(n_jobs=n_jobs, batch_size=batch_size, **parallel_para_dict)
        func = delayed(func)

        if respective:
            if respective_kwargs:
                return para_func(func(*iter_i, **kws, **kwargs) for *iter_i, kws in tqdm2(iterable, tq=tq, desc=desc))
            else:
                return para_func(func(*iter_i, **kwargs) for iter_i in tqdm2(iterable, tq=tq, desc=desc))
        else:
            return para_func(func(iter_i, **kwargs) for iter_i in tqdm2(iterable, tq=tq, desc=desc))

    elif mode in ["imap", "im"]:

        if not respective:
            func2 = partial(func, **kwargs)
        else:
            func2 = partial(funcz, ff=func, respective=respective, respective_kwargs=respective_kwargs, **kwargs)

        pool = multiprocessing.Pool(processes=n_jobs)

        result = tqdm2(pool.imap(func=func2, iterable=iterable, chunksize=chunksize), desc=desc, tq=tq)
        # This is just used in Python mode rather than IPython

        if not lazy:
            result = [result_i for result_i in result]
            pool.close()
            pool.join()
            return result
        else:
            """## lazy result: please make sure run the following code after get result.
            >>> result = [for result_i in result] # in this line, you could use you function to deal with 'result_i'.
            >>> pool.close()
            >>> pool.join()
            """
            return result, pool

    elif mode in ["map", "starmap", "starmap_async", "map_async"]:
        assert respective_kwargs is False
        func = partial(func, **kwargs)
        pool = multiprocessing.Pool(processes=n_jobs)

        if lazy:
            assert "async" in mode, 'Just accept (lazy==True,mode=="**_async") or (lazy==False,mode=="**")'
        else:
            assert "async" not in mode, 'Just accept (lazy==True,mode=="**_async") or (lazy==False,mode=="**")'

        if respective:
            assert "star" in mode, 'If respective==True and mode in map class, mode could be "starmap" or ' \
                                   '"starmap_async") '
        else:
            assert "star" not in mode, 'If respective==False and mode in map class, mode could be "map" or "map_async")'

        para_func = getattr(pool, mode)

        if not lazy:
            result = tqdm2(para_func(func=func, iterable=iterable), desc=desc, tq=tq)
            result = [result_i for result_i in result]
            pool.close()
            pool.join()
            return result
        else:
            result = para_func(func=func, iterable=iterable)
            result = (result_i for result_i in tqdm2(result.get(), tq=tq, desc=desc))
            """## lazy result: please make sure run the following code after get result.
            >>> result = [for result_i in result] # in this line, you could use you function to deal with 'result_i'.
            >>> pool.close()
            >>> pool.join()
            """
            return result, pool

    elif mode in ["apply", "apply_async"]:

        func = partial(func, **kwargs)
        pool = multiprocessing.Pool(processes=n_jobs)

        if lazy and mode == "apply_async":
            para_func = pool.apply_async
        elif not lazy and mode == "apply":
            para_func = pool.apply
        else:
            raise KeyError('Just accept (lazy==True,mode=="apply_async") or (lazy==False,mode=="apply")')

        if respective:
            if respective_kwargs:
                result = [para_func(func=func, args=iter_i, kwds=kwds)
                          for *iter_i, kwds in tqdm2(iterable, tq=tq, desc=desc)]
            else:
                result = [para_func(func=func, args=iter_i) for (*iter_i,) in tqdm2(iterable, tq=tq, desc=desc)]
        else:
            result = [para_func(func=func, args=(iter_i,)) for iter_i in tqdm2(iterable, tq=tq, desc=desc)]

        if not lazy:
            pool.close()
            pool.join()
            return result
        else:
            result = (result_i.get() for result_i in tqdm2(result, tq=tq, desc=desc))
            """## lazy result: please make sure run the following code after get result.
            >>> result = [for result_i in result] # in this line, you could use you function to deal with 'result_i'.
            >>> pool.close()
            >>> pool.join()
            """
            return result, pool

    else:
        raise NameError('Accept mode: ["apply", "apply_async", "map", "starmap",'
                        '"starmap_async", "map_async", "imap", "im", "j", "joblib"]')


def batch_parallelize(n_jobs, func, iterable, respective=False, tq=True, batch_size: int = 1000, store=None, mode="j",
                      parallel_para_dict: dict = None, respective_kwargs=False, desc=None, lazy=False,
                      **kwargs):
    """
    Parallelize the function for iterable.

    The iterable would be batched into batch_size  for less resource's transmission.

    Examples
    ----------
    >>> def func(x):
    ...     return x**2
    >>> if __name__=="__main__":
    >>>     result = parallelize(n_jobs=2,func=func,iterable=[1,2,3,4,5])

    [1,2,3,4,5]

    Note:
    For "small" calculated function, with small return for each function.

    def func:
        'small' calculation.
        ...
        return int

    make sure the if __name__ == "__main__":

    Parameters
    ----------
    respective_kwargs:
        the respective parameters contains kwargs or not. only for mode=="j" and respective=True.
        the first iterable i is tuple and second kw is dict.
        # >>> for iter_i, kw in tqdm(iterable)):
        # >>>    func(*iter_i,**kw)

    parallel_para_dict:dict
        Parameters passed to joblib.Parallel
    batch_size:int,str
        For small data and very big loop.with model "m"
    respective:bool
        Import the parameters respectively or as a whole
    tq:bool
         View progress or not
    n_jobs:int
        cpu numbers. n_jobs is the number of workers requested by the callers. Passing n_jobs=-1
        means requesting all available workers for instance matching the number of CPU cores on the worker host(s).
    func:
        function to calculate
    iterable:
        iterable object
    mode:
        "j":"joblib" or "m":"multiprocessing"
        m for very big data and small loop.
    kwargs:
        kwargs for function
    desc:str
        Prefix for the progressbar.
    store:bool,None
        Not used, store or not, if store, the result would be store to disk and return nothing.
    lazy: bool
        return generator (lazy=True) or result list (lazy=False).

    Returns
    -------
    results:
        function results

    """
    if respective_kwargs:
        assert respective is True

    if parallel_para_dict is None:
        parallel_para_dict = {}

    if effective_n_jobs(n_jobs) == 1:
        return parallelize(n_jobs, func=func, iterable=iterable, respective=respective, tq=tq,
                           batch_size=batch_size, store=store, desc=desc, lazy=lazy,
                           respective_kwargs=respective_kwargs, parallel_para_dict=parallel_para_dict,
                           **kwargs, )

    iterable = list(iterable)
    batch = len(iterable) // batch_size + 1
    iterable = np.array(iterable, dtype=object)
    iterables = np.array_split(iterable, batch)

    if mode in ["m", "imap", "im"]:

        pool = multiprocessing.Pool(processes=n_jobs)

        func2 = partial(funcz_batch, ff=func, respective=respective, respective_kwargs=respective_kwargs, **kwargs)

        rett = tqdm2(pool.imap(func=func2, iterable=iterables),
                     total=len(iterables), desc=desc, tq=tq)
        pool.close()
        pool.join()

        result = []
        if not lazy:
            [result.extend(i) for i in rett]
            pool.close()
            pool.join()
            return result
        else:
            return itertools.chain(*rett), pool

    elif mode in ["j", "joblib"]:
        assert lazy is False

        func2 = partial(funcz_batch, ff=func, respective=respective, respective_kwargs=respective_kwargs, **kwargs)

        parallel = Parallel(n_jobs=n_jobs, batch_size=1, **parallel_para_dict)

        func_batch = delayed(func2)

        try:
            y = parallel(func_batch(iter_i) for iter_i in tqdm2(iterables, desc=desc, tq=tq))
            ret = []
            [ret.extend(i) for i in y]
            return ret
        except MemoryError:
            raise MemoryError(
                "The total size of calculation is out of Memory, please spilt your data.")
    else:
        raise NameError('Accept mode: ["imap", "im", "j", "joblib"]')


def parallelize_imap(n_jobs, func, iterable, tq=True, desc=None):
    """
    Parallelize the function for iterable.

    For very big loop and small data.

    Parameters
    ----------
    func:function
    iterable:List
    n_jobs:int
    tq:bool
    desc:str
        Prefix for the progressbar.
    """
    # This is just used in Python mode rather than IPython
    pool = multiprocessing.Pool(processes=n_jobs)
    if tq:
        result_list_tqdm = [result for result in tqdm(pool.imap(func=func, iterable=iterable),
                                                      total=len(iterable), desc=desc)]
        pool.close()
        pool.join()
    else:
        result_list_tqdm = [result for result in pool.imap(func=func, iterable=iterable)]
        pool.close()
        pool.join()

    return result_list_tqdm


def parallelize_parameter(func, iterable, respective=False, **kwargs):
    import multiprocessing
    maxx = multiprocessing.cpu_count()
    n_jobs = maxx - 2

    batch_size = [5, 10, 25, 50, 100]

    iterable2 = []
    for i, j in enumerate(iterable):
        iterable2.append(j)
        if i > 10:
            break
    iterable2 = [iterable2] * 300
    iterable2 = list(chain(*iterable2))

    t1 = time.time()
    if respective:
        a = list(func(*iter_i) for iter_i in iterable2[:16])
    else:
        a = list(func(iter_i) for iter_i in iterable2[:16])
    t2 = time.time()
    t12 = (t2 - t1) / 16
    size = sum([getsizeof(i) for i in a]) / 16 / 1024
    print("Time of calculation/each:%s," % t12, "each output size:%s Kbytes" % size)

    for batch_sizei in batch_size:
        t3 = time.time()
        parallelize(n_jobs, func, iterable2, respective, batch_size=batch_sizei, **kwargs)
        t4 = time.time()
        t34 = t4 - t3
        size2 = size * batch_sizei
        print("Total time/each:%s," % (t34 / 300), "Return output time/each:%s," % (t34 / 300 - t12),
              "Batch output size:%s Kbytes," % size2, "Batch_size:%s" % batch_sizei)

    print("Choice the batch_size with min Total time.")


def logg(func, printing=True, back=False):
    """
    Get the name of function.
    use as a decorator:@

    Parameters
    ----------
    func:Callable
        function to calculate
    printing:bool
        print or not
    back:bool
        return result or not

    Returns
    -------
    function results
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if inspect.isclass(func):
            name = "instance of %s" % func.__name__
            arg_dict = ""
            if printing:
                print(name, arg_dict)
            result = func(*args, **kwargs)
        elif inspect.isfunction(func):
            arg_dict = inspect.getcallargs(func, *args, **kwargs)
            name = func.__name__
            if printing:
                print(name, arg_dict)
            result = func(*args, **kwargs)
        else:
            arg_dict = ""
            name = ""
            result = func(*args, **kwargs)
            pass
        if back:
            return (name, arg_dict), result
        else:
            return result

    return wrapper


def name_to_name(*iters, search, search_which=1, return_which=(1,), two_layer=False):
    """
    Search and rank the list.

    Parameters
    ----------
    iters:tuple
        iterable objects to select and sort
    search:Iterable
        the rank basis
    search_which:int
        the (the index of iters) of the rank basis. where to rank
    return_which:tuple of int
        return index in inters
    two_layer:bool
        search is nested with two layer or not

    Returns
    -------
    result:Iterable
        Result of find and sort
    """
    if isinstance(return_which, int):
        return_which = tuple([return_which, ])
    if two_layer:

        results_all = []
        if isinstance(search, Iterable):
            for index_i in search:
                results_all.append(
                    name_to_name(*iters, search=index_i, search_which=search_which,
                                 return_which=return_which, two_layer=False))

            if len(return_which) >= 2:
                return list(zip(*results_all))
            else:
                return results_all
        else:
            raise IndexError("search_name or search should be iterable")

    else:

        zeros = [list(range(len(iters[0])))]

        zeros.extend([list(_) for _ in iters])

        iters = zeros

        zips = list(zip(*iters))

        if isinstance(search, Iterable):

            search_index = [iters[search_which].index(i) for i in search]

            results = [zips[i] for i in search_index]

        else:

            raise IndexError("search_name or search should be iterable")

        res = list(zip(*results))

        if not res:
            return_res = [[] for _ in return_which]
        else:
            return_res = [res[_] for _ in return_which]

        if len(return_which) == 1:
            return_res = return_res[0]
        return return_res


class _TTClass(dict):

    def __init__(self, **kwargs):
        super(_TTClass, self).__init__(**kwargs)

    def _t(self):
        self._r()

    def _p(self):
        a = np.array(list(self.keys()))
        b = np.array(list(self.values()))
        a0 = np.delete(a, 0)
        a = np.delete(a, -1)
        b0 = np.delete(b, 0)
        b = np.delete(b, -1)
        ti = b0 - b

        def func(x, y):
            return "{}-{}:".format(x, y)

        ufunc = np.frompyfunc(func, 2, 1)
        ni = ufunc(a0, a)
        re = np.vstack((ni, ti))
        re = re.T
        print(re)
        self.clear()

    def _r(self, name=None):
        ti = time.time()
        if name is None:
            n = len(self)
            name = "t%s" % n
        self[name] = ti

    def subs(self, name1, name2):
        if name2 in self and name2 in self:
            return self[name1] - self[name2]
        else:
            raise NameError("There is no name:{} or {}".format(name1, name2))

    # def tp(self, name=None):
    #     ti = time.time()
    #     n = len(self)
    #     if name is None:
    #         name = "tp"
    #     if n >= 1:
    #         name0 = list(self.keys())[-1]
    #         t0 = list(self.values())[-1]
    #         print("-".join((name,name0)), ti - t0)
    #     else:
    #         raise UserWarning("The .tp only used after .t, or .r")


class TTClass(_TTClass):
    """
    Quick time. Don't use this class use 'tt' object.

    Examples:
    -----------
    >>> from mgetool.tool import tt
    >>> tt.t
    >>> a=4
    >>> tt.t
    >>> tt.p
    [['t1-t0:' ****]]

    """

    def __init__(self, **kwargs):
        super(_TTClass, self).__init__(**kwargs)

    def __getattribute__(self, item):
        if item == "t":
            _TTClass._t(self)

        elif item == "p":
            _TTClass._p(self)

        elif item[-1] in "0123456789":
            _TTClass._r(self, name=item)

        else:
            return _TTClass.__getattribute__(self, item)


def def_pwd(path=None, change=False, verbose=False):
    """try of get and define work path."""
    if path is None:
        path = os.getcwd()
        pwd = path
    if os.path.exists(path):
        path = os.path.abspath(path)
        if change:
            os.chdir(path)
        pwd = os.getcwd()
    else:
        os.makedirs(path)
        path = os.path.abspath(path)
        if change:
            os.chdir(path)
        pwd = os.getcwd()
    if verbose:
        print("work path:", pwd)
        print("checked path:", path)
    locals()[pwd] = pwd
    return path


def cmd_sys(d, cmd):
    """Run linux cmd"""
    old = os.getcwd()
    os.chdir(d)
    os.system(cmd)
    os.chdir(old)


def cmd_popen(d, cmd):
    """Run linux cmd and return result."""
    old = os.getcwd()
    os.chdir(d)
    res = os.popen(cmd).readlines()
    os.chdir(old)
    return res


tt = TTClass()


def dos2unix(file, out_file=None):
    """

    Args:
        file: (str,) input file name
        out_file: (str,) input file name, if None, cover the input.
    """
    with open(file, 'rb') as infile:
        content = infile.read()
    out_file = file if out_file is None else out_file
    with open(out_file, 'wb') as output:
        for line in content.splitlines():
            output.write(line + b'\n')


if __name__ == "__main__":
    def func(n, z=4
             ):
        time.sleep(0.000001)
        s = np.random.random((100, 50)) ** 2
        return s


    iterable = np.arange(1000)
    iterable2 = np.arange(2000)
    # tt.t
    # s = parallelize(1, func, zip(iterable, iterable), respective=True, tq=True, batch_size=50)
    # tt.t
    # s = parallelize(4, func, zip(iterable, iterable), respective=True, tq=True, batch_size=50)
    # tt.t
    # s = parallelize(1, func, iterable, respective=False, tq=True, batch_size=50)
    # tt.t
    # s = parallelize(4, func, iterable, respective=False, tq=True, batch_size=50)

    # tt.t
    # s = batch_parallelize(1, func, zip(iterable, iterable), respective=True, tq=True, batch_size=100, )
    # tt.t
    # s = batch_parallelize(2, func, zip(iterable, iterable), respective=True, tq=True, batch_size=100, )
    #
    # tt.t
    # s = batch_parallelize(1, func, iterable, respective=False, tq=True, batch_size=100, mode="m")
    # tt.t
    # s = batch_parallelize(4, func, iterable, respective=False, tq=True, batch_size=100, mode="m")
    # tt.t
    # tt.p
