#include <cstdlib>
#include <iostream>
#include <matplotlibcpp.hpp>
#include <vector>
using namespace std;
namespace plt = matplotlibcpp;
int main() {
    vector<int> x, y;
    for (int i = 1; i < 30; i++) {
        y.push_back(rand() % 100);
        x.push_back(i);
    }
    plt::plot(x, y);
    plt::show();
}
