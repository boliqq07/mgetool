#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/29 19:47
# @Author : DEAP
# @Software: PyCharm
# @License: BSD 3-Clause

"""
Notes:
    This file is part of DEAP.
    this is used for pack some function into one box
"""

from inspect import getfullargspec

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from copy import deepcopy
from functools import partial, wraps


class Toolbox(object):
    """A toolbox for evolution that contains the evolutionary operators. At
    first the toolbox contains a :meth:`~deap.toolbox.clone` method that
    duplicates any element it is passed as argument, this method defaults to
    the :func:`copy.deepcopy` function. and a :meth:`~deap.toolbox.map`
    method that applies the function given as first argument to every items
    of the iterables given as next arguments, this method defaults to the
    :func:`map` function. You may populate the toolbox with any other
    function by using the :meth:`~deap.base.Toolbox.register` method.

    Concrete usages of the toolbox are shown for initialization in the
    :ref:`creating-types` tutorial and for tools container in the
    :ref:`next-step` tutorial.
    """

    def __init__(self):
        self.register("clone", deepcopy)
        self.register("map", map)

    def register(self, alias, function, *args, **kargs):
        """Register a *function* in the toolbox under the name *alias*. You
        may provide default arguments that will be passed automatically when
        calling the registered function. Fixed arguments can then be overriden
        at function call time.

        :param alias: The name the operator will take in the toolbox. If the
                      alias already exist it will overwrite the the operator
                      already present.
        :param function: The function to which refer the alias.
        :param args: One or more argument (and keyword argument) to pass
                         automatically to the registered function when called,
                         optional.

        The following code block is an example of how the toolbox is used. ::

            def func(a, b, c=3):
                    pass
            tools = Toolbox()
            tools.register("myFunc", func, 2, c=4)
            tools.myFunc(3)


        The registered function will be given the attributes :attr:`__name__`
        set to the alias and :attr:`__doc__` set to the original function's
        documentation. The :attr:`__dict__` attribute will also be updated
        with the original function's instance dictionary, if any.
        """
        pfunc = partial(function, *args, **kargs)
        pfunc.__name__ = alias
        pfunc.__doc__ = function.__doc__

        if hasattr(function, "__dict__") and not isinstance(function, type):
            # Some functions don't have a dictionary, in these cases
            # simply don't copy it. Moreover, if the function is actually
            # a class, we do not want to copy the dictionary.
            pfunc.__dict__.update(function.__dict__.copy())

        setattr(self, alias, pfunc)

    def unregister(self, alias):
        """Unregister *alias* from the toolbox.

        :param alias: The name of the operator to remove from the toolbox.
        """
        delattr(self, alias)

    def decorate(self, alias, *decorators):
        """Decorate *alias* with the specified *decorators*, *alias*
        has to be a registered function in the current toolbox.

        :param alias: The name of the operator to decorate.
        :param decorators: One or more function decorator. If multiple
                          decorators are provided they will be applied in
                          order, with the last decorator decorating all the
                          others.

        .. note::
            Decorate a function using the toolbox makes it unpicklable, and
            will produce an error on pickling. Although this limitation is not
            relevant in most cases, it may have an impact on distributed
            environments like multiprocessing.
            A function can still be decorated manually before it is added to
            the toolbox (using the @ notation) in order to be picklable.
        """
        pfunc = getattr(self, alias)
        function, args, kargs = pfunc.func, pfunc.args, pfunc.keywords
        for decorator in decorators:
            function = decorator(function)
        self.register(alias, function, *args, **kargs)

    def refresh(self, alias=None, *nargs, **nkwargs):
        "the refreshed function  only can be used by kwargs parameter ranther args "

        if isinstance(alias, str):
            if hasattr(self, alias):
                pfunc = getattr(self, alias)
                function, args, kargs = pfunc.func, pfunc.args, pfunc.keywords

                detail = getfullargspec(function)

                n_arg = len(detail.args)
                if detail.defaults:
                    defu = [None] * (n_arg - len(detail.defaults))
                    defu.extend(detail.defaults)  # ?
                else:
                    defu = [None] * n_arg

                defu_dict = {}
                for i, j in zip(detail.args, defu):
                    if j is not None:
                        defu_dict[i] = j

                if detail.kwonlydefaults:
                    defu_dict.update(detail.kwonlydefaults)

                for i, j in zip(detail.args, args):
                    defu_dict[i] = j

                defu_dict.update(kargs)

                defu_dict.update(nkwargs)

                for i, j in zip(detail.args, nargs):
                    if i in defu_dict:
                        del defu_dict[i]

                self.register(alias, function, *nargs, **defu_dict)
            else:
                pass

        elif isinstance(alias, (list, tuple)):
            for i in alias:
                self.refresh(alias=i, *nargs, **nkwargs)


if __name__ == "__main__":
    def func(a, b=1, c=2, name=4, **kwargs):
        print(a, b, c, name, kwargs)
        pass


    def staticLimit(key, max_value):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                new_inds = func(*args, **kwargs)

                print(key, max_value)
                return new_inds

            return wrapper

        return decorator


    to = Toolbox()

    to.register("a", func, 1, 5, abss=3)
    to.decorate("a", staticLimit(key=4, max_value=5))
    to.refresh("a", 3, 4, 6, name=3, abss=4)
    to.a()
