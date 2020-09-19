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
    // data = std::vector<std::string> 
    auto col = xt::col(xt::view(data, xt::range(1, 5), xt::range(0, 1)), 0);
    std::vector <double> vec(col.size());
    
    for ( int i = 0 ; i < col.size() ; i++ ){
        vec.push_back(std::stod(col[i]));
    }
    matplotlibcpp::plot(vec);
    matplotlibcpp::show();
    return 0;
}
