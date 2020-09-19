#include <fstream>
#include <iostream>
#include <istream>
#include <string>

#include <matplotlibcpp.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xoperation.hpp>

int main() {
    std::ifstream in_file;
    in_file.open("./datasets/boston/boston.csv");
    auto data = xt::load_csv<std::string>(in_file);
    auto stringVector = xt::col(xt::view(data, xt::range(1, 5), xt::range(0, 1)), 0);
    std::vector<std::string> str("0.123", "0.232", "0.617");
    std::vector<double> doubleVector(stringVector.size());
    std::transform( str.begin(), str.end(), std::back_inserter(doubleVector), [](std::string const& val) {return std::stod(val);});
    for ( int i = 0 ; i < 5 ; i++ ){
        std::cout<<doubleVector[i]<<" ";
    }
    return 0;
}