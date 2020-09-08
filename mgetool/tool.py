#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/29 19:44
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
Notes:
    Some tools for characterization
"""
import inspect
import numbers
import random
import time
from collections.abc import Iterable
from functools import partial, wraps

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from tqdm import tqdm


def time_this_function(func):
    """
    time the function.
    use as a decorator.

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
    if use the numpy random, please user the check_random_state in sklearn.

    Parameters
    ----------
    seed: None,int or RandomState
        If seed is None, return the RandomState singleton used by random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

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


def parallelize(n_jobs, func, iterable, respective=False, tq=True, batch_size='auto', **kwargs):
    """
    parallelize the function for iterable.

    make sure in if __name__ == "__main__":

    Parameters
    ----------
    batch_size
    respective:bool
        Import the parameters respectively or as a whole
    tq:bool
         View Progress or not
    n_jobs:int
        cpu numbers. n_jobs is the number of workers requested by the callers. Passing n_jobs=-1
    means requesting all available workers for instance matching the number of CPU cores on the worker host(s).
    func:
        function to calculate
    iterable:
        interable object
    kwargs:
        kwargs for function

    Returns
    -------
    results
        function results

    """

    func = partial(func, **kwargs)
    if effective_n_jobs(n_jobs) == 1:
        parallel, func = list, func
    else:
        parallel = Parallel(n_jobs=n_jobs, batch_size=batch_size)
        func = delayed(func)
    if tq:
        if respective:
            return parallel(func(*iter_i) for iter_i in tqdm(iterable))
        else:
            return parallel(func(iter_i) for iter_i in tqdm(iterable))
    else:
        if respective:
            return parallel(func(*iter_i) for iter_i in iterable)
        else:
            return parallel(func(iter_i) for iter_i in iterable)


def logg(func, printing=True, back=False):
    """
    Get the name of function
    use as a decorator

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
    search and rank the list.

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
    quick time.
    use tt object.
    >>>tt.t
    >>>#tt.t
    >>>tt.p
    """

    def __init__(self, **kwargs):
        super(_TTClass, self).__init__(**kwargs)

    def __getattribute__(self, item):
        if item is "t":
            _TTClass._t(self)

        elif item is "p":
            _TTClass._p(self)

        elif item[-1] in "0123456789":
            _TTClass._r(self, name=item)

        else:
            return _TTClass.__getattribute__(self, item)


tt = TTClass()
