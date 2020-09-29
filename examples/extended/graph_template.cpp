#include <cstdlib>
#include <iostream>
#include <matplotlibcpp.hpp>
#include <vector>

int main() {
    std::vector<int> x, y;
    for (int i = 1; i < 30; i++) {
        y.push_back(rand() % 100);
        x.push_back(i);
    }

    matplotlibcpp::plot(x, y, "ro");
    matplotlibcpp::show();
}
