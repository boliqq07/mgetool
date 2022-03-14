from mgetool.tool import parallelize, tt, parallelize_imap, batch_parallelize
import numpy as np


def func(n, _=None):
    # time.sleep(0.0001)
    s = np.random.random((10, 50))
    return s


if __name__ == "__main__":
    iterable = np.arange(50)
    s0 = batch_parallelize(4, func, iterable, respective=False, tq=True, mode="m")  # 无tq
    s1 = parallelize(4, func, iterable, respective=False, tq=True, mode="j")
    s2 = parallelize_imap(4, func, iterable, tq=True)
    s0 = parallelize(4, func, iterable, respective=False, tq=False, mode="m")  # 无tq
    s1 = parallelize(4, func, iterable, respective=False, tq=False, mode="j")
    s2 = parallelize_imap(4, func, iterable, tq=False)


    def func(n, _=None):
        # time.sleep(0.0001)
        s = np.random.random((100, 50))
        return s


    iterable = np.arange(10000)

    print("samll loop and big data")

    tt.t
    s0 = batch_parallelize(4, func, iterable, respective=False, tq=True, batch_size=500, mode="m")
    tt.t
    s1 = parallelize(4, func, iterable, respective=False, tq=True, mode="m")
    tt.t
    tt.p
