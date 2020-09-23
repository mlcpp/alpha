#include <Matrix.hpp>
#include <preprocessing.hpp>

/* Example program

Read csv files to get a Matrix object.
Slice the Matrix objects such that it can be converted to double.
Normalize the Matrix object using normalize method and printed the normalized matrix.
*/
int main() {
    Matrix mat = read_csv("./datasets/boston/boston.csv");

    Matrix sliced_mat = mat.slice(1, mat.row_length(), 0, mat.col_length());
    sliced_mat.to_double();

    // Normalizing the Matrix object
    Matrix normalized = preprocessing.normalize(sliced_mat, "row");
    normalized.print();

    return 0;
}