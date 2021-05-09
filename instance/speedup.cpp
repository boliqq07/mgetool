//
// Created by Report on 2021/2/6.
//
#include <vector>
#include <array>
#include <iostream>

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
