import os

from mgetool.draft import DraftPyx

# bd = DraftPyx("./new/hello.pyx",language="c++")
# print(os.getcwd())
# bd.write()
# print(os.getcwd())
# bd.quick_import(build=True, with_html=True)
# print(os.getcwd())
#
# bd.remove()


# from mgetool.draft import DraftPybind11
#
# bd = DraftPybind11("speedup_pybind11.cpp")
# bd.write(functions=["cubic", "fcc", "bcc"])
# a = bd.quick_import(build=True, with_html=False)
# bd.remove()

# from mgetool.draft import DraftTorch
#
# bd = DraftTorch("speedup_torch.cpp")
# bd.write(functions=["d_sigmoid",])
# a= bd.quick_import(build=True)
# bd.remove()


# from mgetool.draft import TorchJit
#
# bd = TorchJit("speedup_torch.cpp")
# bd.write(functions=["d_sigmoid",])
# a= bd.quick_import(build=True)
# bd.remove()

from mgetool.draft import TorchJitInLine

source = """
#include <torch/extension.h>
#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}
"""
bd = TorchJitInLine(source=source)
bd.write(functions=["d_sigmoid", ])
a = bd.quick_import(build=True)
# bd.remove()

from featurebox.featurizers import *
