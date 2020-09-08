# MGE tool
Some useful base tools for other mge packages, but without specific task.

[![Python Versions](https://img.shields.io/pypi/pyversions/mgetool.svg)](https://pypi.org/project/mgetool/)
[![Version](https://img.shields.io/github/tag/boliqq07/mgetool.svg)](https://github.com/boliqq07/mgetool/releases/latest)
![pypi Versions](https://badge.fury.io/py/mgetool.svg)

# Install
```bash
pip install mgetool
```

# contains

export
----------------------
**Store**

Store files in batches.(无差别存储数据)

import
----------------------
**Call**

Import files in batches.(无差别导入数据)

show
----------------------
**BasePlot**

Draw picture quickly.(快速画图)

**corr_plot**

Draw correlation coefficient graph.(相关系数图)

tool
----------------------
**tt**
```python
from mgetool.tool import tt
tt.t
...
tt.t
tt.p
```
Record the time of this site, and print uniformly.(测试代码块时间)

**time_this_function**

Time the function(测试函数运行时间，@)

**parallelize**

Parallelize the **for** loop (并行化)

**logg**

Get the name of function(输出函数信息，@)

newclass
----------------------

```python
from mgetool.newclass import create

import numpy as np
def ff(x, y):
    print(y * 1000)
    return x

Foo = create("Foo", np.ndarray, lenn=dict, spam=1, fu=ff)
foo = Foo([1, 2, 3])
a = foo.fu(2, 4)
```
Build a simple class quickly.(No initialization parameters)（快速创建新类）

packbox
----------------------
```python
from mgetool.packbox import Toolbox

def func(a, b=1, c=2, name=4, **kwargs):
    print(a, b, c, name, kwargs)
    pass


to = Toolbox()

to.register("a", func, 1, 5, abss=3)
to.refresh("a", 3, 4, 6, name=3, abss=4)
to.a()
```
Build a toolbox and you can add function to it.(函数集合)
