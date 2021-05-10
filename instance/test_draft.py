# from mgetool.draft import DraftPyx
#
# bd = DraftPyx("hello.pyx")
# bd.write()
# a= bd.quick_import(build=True, with_html=True)
# bd.remove()


from mgetool.draft import DraftPybind11

bd = DraftPybind11("mgetool_pybind11.cpp")
bd.write(functions=["cubic", "fcc", "bcc"])
a = bd.quick_import(build=True, with_html=False)
bd.remove()

# from mgetool.draft import DraftTorch
#
# bd = DraftTorch("mgetool_torch.cpp")
# bd.write(functions=["d_sigmoid",])
# a= bd.quick_import(build=True)
# bd.remove()


# from mgetool.draft import TorchJit
#
# bd = TorchJit("mgetool_torch.cpp")
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
a = bd.quick_import(build=False)
# bd.remove()

from featurebox.featurizers import *
