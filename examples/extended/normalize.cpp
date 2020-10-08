#include <Matrix.hpp>
#include <matplotlibcpp.hpp>
#include <preprocessing.hpp>
namespace plt = matplotlibcpp;

// Example program

// Read csv files to get a Matrix object.
// Slice the Matrix object according to our needs.
// Normalize the Matrix object using normalize() method.
// The difference between normalized and unnormalized datasets is visually depicted.
int main() {
    Matrix mat = read_csv("./datasets/blobs_norm/blobs_norm.csv");
    mat = mat.slice(1, mat.row_length(), 0, mat.col_length() - 1);
    mat.to_double();

    plt::figure_size(800, 600);
    plt::plot(mat.get_col(0), mat.get_col(1), "ro");
    plt::title("Unnormalized");
    plt::show();

    // Normalizing the Matrix object
    Preprocessing preprocessing;
    Matrix normalized = preprocessing.normalize(mat, "column");

    plt::figure_size(800, 600);
    plt::plot(normalized.get_col(0), normalized.get_col(1), "ro");
    plt::title("Normalized");
    plt::show();

    return 0;
}