#include <fstream>
#include <iostream>
#include <istream>

#include "xtensor/xarray.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

int main() {
    std::ifstream in_file;
    in_file.open("../datasets/boston/boston.csv");
    auto data = xt::load_csv<std::string>(in_file);

    std::cout << xt::view(data, xt::range(0, 5), xt::all()) << std::endl;
    return 0;
}
