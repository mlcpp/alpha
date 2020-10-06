#include <Matrix.hpp>
#include <matplotlibcpp.hpp>
namespace plt = matplotlibcpp;

int main() {
    // Specify backend renderer for matplotlib
    plt::backend("GTK3Agg");
    // Load the dataset
    Matrix mat = read_csv("./datasets/blobs_pca/blobs_pca.csv");

    // Remove the header description from the dataset
    Matrix X = mat.slice(1, mat.row_length(), 0, mat.col_length());

    // Convert the data from string to double
    X.to_double();

    plt::figure_size(800, 600);
    plt::plot(X.get_col(0), X.get_col(1), "ro");
    plt::title("Example Graph");
    plt::show();
}
