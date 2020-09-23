#include <Matrix.hpp>
#include <preprocessing.hpp>

/* Example program

Read csv files to get a Matrix object.
Slice the Matrix object according to our needs.
Normalize the Matrix object using normalize method and printed the normalized matrix.
*/
int main() {
    Matrix mat = read_csv("./datasets/boston/boston.csv");

    Matrix sliced_mat = mat.slice(1, 6, 2, 5);
    sliced_mat.to_double();

    // Normalizing the Matrix object
    Matrix normalized = preprocessing.normalize(sliced_mat, "row");
    normalized.print();

    return 0;
}