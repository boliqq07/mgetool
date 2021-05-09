//
// Created by Report on 2021/2/6.
//
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace Eigen;
using namespace std;

vector<std::array<float, 3>> cubic() {
    vector<std::array<float, 3>> kpcubic = { /* NOLINT */
            std::array<float, 3>({0.0, 0.0, 0.0}),
            std::array<float, 3>({0.0, 0.5, 0.0}),
            std::array<float, 3>({0.5, 0.5, 0.5}),
            std::array<float, 3>({0.5, 0.5, 0.0}),
    };
    return kpcubic;
}

vector<std::array<float, 3>> fcc() {
    vector<std::array<float, 3>> kpfcc = { /* NOLINT */
            std::array<float, 3>({0.0, 0.0, 0.0}),
            std::array<float, 3>({3.0 / 8.0, 3.0 / 8.0, 3.0 / 4.0}),
            std::array<float, 3>({0.5, 0.5, 0.5}),
            std::array<float, 3>({5.0 / 8.0, 1.0 / 4.0, 5.0 / 8.0}),
            std::array<float, 3>({0.5, 1.0 / 4.0, 3.0 / 4.0}),
            std::array<float, 3>({0.5, 0.0, 0.5}),
    };
    return kpfcc;
}

vector<std::array<float, 3>> bcc() {
    vector<std::array<float, 3>> kpbcc = { /* NOLINT */
            std::array<float, 3>({0.0, 0.0, 0.0}),
            std::array<float, 3>({0.5, -0.5, 0.5}),
            std::array<float, 3>({0.25, 0.25, 0.25}),
            std::array<float, 3>({0.0, 0.0, 0.5}),
    };
    return kpbcc;
}

//# Added
PYBIND11_MODULE(speedup_pybind11, m) {
    m.doc() = "module Document"; // optional module docstring
    m.def("fcc", &fcc, "function Document"); // functions
    m.def("bcc", &bcc, "function Document"); // functions
    m.def("cubic", &cubic, "function Document"); // functions
    }