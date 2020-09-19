#ifndef _matrix_operations_hpp_
#define _matrix_operations_hpp_

#include <all.h>
#include <matrix.hpp>

class MatrixOp {
  private:
    Matrix obj;

  public:
    Matrix read_csv(std::string);
    void concat(std::vector<std::vector<std::string>>);
};

// Method to read a csv file and return a Matrix object
Matrix MatrixOp::read_csv(std::string filename) {
    std::ifstream file(filename);
    std::string line, cell;
    std::vector<std::string> cells;
    char delim = ','; // Delimeter is set to ',' but can be extended

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        while (std::getline(ss, cell, delim)) {
            cells.push_back(cell);
        }
        obj.str_mat.push_back(cells);
        cells.clear();
    }

    return obj;
}

#endif /* _matrix_operations_hpp_ */