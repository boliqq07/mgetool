def myfunction(x, y=2):
    a = x - y
    return a + x * y


def _helper(a):
    return a + 1


class A:
    def __init__(self, b=0):
        self.a = 3
        self.b = b

    def foo(self, x):
        for i in range(x * 1000):
            _ = x
