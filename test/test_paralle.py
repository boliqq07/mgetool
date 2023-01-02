import unittest

import numpy as np

from mgetool.tool import parallelize, parallelize_imap, batch_parallelize


def func(n, l=0,z=0, k=0):
    # time.sleep(0.0001)
    s = np.random.random((100, 50))*l*z*k*n
    return s

iterable = np.arange(1000)
iterable2 = np.arange(1000,2000)
iterable3 = [{"z":10}]*1000


class MyTestCase(unittest.TestCase):


    def test_imap(self):

        ss = parallelize_imap(2, func, iterable, tq=False, desc="sd")
        ss = parallelize_imap(2, func, iterable, tq=True, desc="sd")

    def test_1(self):
        s = parallelize(1, func, zip(iterable, iterable2, iterable3), respective=True,
                        respective_kwargs=True, tq=True,k=3)

        s = parallelize(1, func, zip(iterable, iterable2), respective=True,
                        respective_kwargs=False, tq=False)

        s = parallelize(1, func, zip(iterable, iterable2), respective=True, tq=True,desc="asf")

        s = parallelize(1, func, iterable, respective=False, tq=True)

        ss, pool = parallelize(1, func, zip(iterable, iterable2), respective=True, tq=True,lazy=True)

        ss = [ssi for ssi in ss]

        pool.close()
        pool.join()


    def test_j(self):

        s = parallelize(2, func, zip(iterable, iterable2, iterable3), respective=True,
                        respective_kwargs=True, tq=True,k=3,mode="j",desc="s")

        s = parallelize(2, func, zip(iterable, iterable2), respective=True,
                        respective_kwargs=False, tq=False, mode="j",desc="s")

        s = parallelize(2, func, iterable, respective=False, tq=True,mode="j",desc="s")

        s = parallelize(2, func, zip(iterable, iterable2), respective=True, tq=True,mode="j",desc="s")

    def test_m(self):

        s = parallelize(2, func, zip(iterable, iterable2, iterable3), respective=True,
                        respective_kwargs=True, tq=True, k=3, mode="im", desc="s")

        s = parallelize(2, func, zip(iterable, iterable2), respective=True,
                        respective_kwargs=False, tq=False, mode="im", desc="s")

        s = parallelize(2, func, zip(iterable, iterable2), respective=True, tq=True,mode="im",desc="s")
        #
        s = parallelize(2, func, iterable, respective=False, tq=True,mode="im",desc="s")
        #
        ss, pool = parallelize(2, func, zip(iterable, iterable2), respective=True,mode="im", tq=True,lazy=True)

        ss = [ssi for ssi in ss]

        pool.close()
        pool.join()

    def test_apply(self):

        s = parallelize(2, func, zip(iterable, iterable2, iterable3), respective=True,
                        respective_kwargs=True, tq=True, k=3, mode="apply",desc="s")

        s = parallelize(2, func, zip(iterable, iterable2), respective=True,
                        respective_kwargs=False, tq=False, mode="apply", desc="s")

        s = parallelize(2, func, zip(iterable, iterable2), respective=True, tq=True,mode="apply",desc="s")

        s = parallelize(2, func, iterable, respective=False, tq=True,mode="apply",desc="s")


        ss, pool = parallelize(2, func, zip(iterable, iterable2, iterable3), respective=True,
                        respective_kwargs=True, tq=True, k=3, mode="apply_async",desc="s",lazy=True)

        ss = [ssi for ssi in ss]

        pool.close()
        pool.join()

        ss, pool = parallelize(2, func, zip(iterable, iterable2), respective=True,
                        respective_kwargs=False, tq=False, mode="apply_async", desc="s",lazy=True)

        ss = [ssi for ssi in ss]

        pool.close()
        pool.join()
        ss, pool = parallelize(2, func, zip(iterable, iterable2), respective=True, tq=True,mode="apply_async",desc="s",lazy=True)

        ss = [ssi for ssi in ss]

        pool.close()
        pool.join()
        ss, pool = parallelize(2, func, iterable, respective=False, tq=True,mode="apply_async",desc="s",lazy=True)


        ss = [ssi for ssi in ss]

        pool.close()
        pool.join()

    def test_map_star(self):

        s = parallelize(2, func, zip(iterable, iterable2), respective=True,
                        respective_kwargs=False, tq=False, mode="starmap", desc="s")

        s = parallelize(2, func, zip(iterable, iterable2), respective=True, tq=True,mode="starmap",desc="s")

        s = parallelize(2, func, iterable, respective=False, tq=True,mode="map",desc="s")

        ss, pool = parallelize(2, func, zip(iterable, iterable2), respective=True,
                        respective_kwargs=False, tq=False, mode="starmap_async", desc="s",lazy=True)

        ss = [ssi for ssi in ss]

        pool.close()
        pool.join()

        ss, pool = parallelize(2, func, zip(iterable, iterable2), respective=True, tq=True,
                               mode="starmap_async", desc="s", lazy=True)

        ss = [ssi for ssi in ss]

        pool.close()
        pool.join()

        ss, pool = parallelize(2, func, iterable, respective=False, tq=True, mode="map_async", desc="s", lazy=True)
        ss = [ssi for ssi in ss]

        pool.close()
        pool.join()


    def test_batch_j(self):

        s = batch_parallelize(2, func, zip(iterable, iterable2, iterable3), respective=True,
                        respective_kwargs=True, tq=True,k=3,mode="j",desc="s",batch_size=30,)

        s = batch_parallelize(2, func, zip(iterable, iterable2), respective=True,
                        respective_kwargs=False, tq=False, mode="j",desc="s",batch_size=30,)

        s = batch_parallelize(2, func, iterable, respective=False, tq=True,mode="j",batch_size=30,desc="s")

        s = batch_parallelize(2, func, zip(iterable, iterable2), respective=True, tq=True,mode="j",desc="s",batch_size=30,)

    def test_batch_m(self):

        s = batch_parallelize(2, func, zip(iterable, iterable2, iterable3), respective=True,batch_size=30,
                        respective_kwargs=True, tq=True, k=3, mode="im", desc="s")

        s = batch_parallelize(2, func, zip(iterable, iterable2), respective=True,batch_size=30,
                        respective_kwargs=False, tq=False, mode="im", desc="s")

        s = batch_parallelize(2, func, zip(iterable, iterable2), respective=True,batch_size=30, tq=True,mode="im",desc="s")
        #
        s = batch_parallelize(2, func, iterable, respective=False, tq=True,mode="im",desc="s",batch_size=30,)
        #
        ss, pool = batch_parallelize(2, func, zip(iterable, iterable2), respective=True,mode="im",batch_size=30,tq=True,lazy=True)

        ss = [ssi for ssi in ss]

        pool.close()
        pool.join()







if __name__ == '__main__':
    unittest.main()
