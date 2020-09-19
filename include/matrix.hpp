#ifndef _matrix_hpp_
#define _matrix_hpp_

#include <all.h>

class Matrix {
  public:
    std::vector<std::vector<std::string>> str_mat;
    std::vector<std::vector<double>> dob_mat;

    void print();
    Matrix slice(int, int, int, int);
    int col_length();
    int row_length();
    void to_double();
};

void Matrix::print() {
    for (int i = 0; i < str_mat.size(); i++) {
        for (int j = 0; j < str_mat[i].size(); j++)
            std::cout << str_mat[i][j] << " ";
        std::cout << std::endl;
    }
}

Matrix Matrix::slice(int row_start, int row_end, int col_start, int col_end) {
    Matrix mat;
    std::vector<std::string> row;

    for (int i = row_start; i < row_end; i++) {
        for (int j = col_start; j < col_end; j++)
            row.push_back(str_mat[i][j]);
        mat.str_mat.push_back(row);
        row.clear();
    }

    return mat;
}

int Matrix::col_length() { return str_mat[0].size(); }
int Matrix::row_length() { return str_mat.size(); }

void Matrix::to_double() {
    std::vector<double> row;
    for (int i = 0; i < str_mat.size(); i++) {
        for (int j = 0; j < str_mat[i].size(); j++)
            row.push_back(std::stod(str_mat[i][j]));
        dob_mat.push_back(row);
        row.clear();
    }
}

#endif /* _matrix_hpp_ */