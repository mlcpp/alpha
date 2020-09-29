// import header files
#include <KMeans.hpp>
#include <matplotlibcpp.hpp>
namespace plt = matplotlibcpp;

// Example program

// Read csv files to get a Matrix object.
// Slice the Matrix object to a suitable size.
// Plot uncategorized data points
// Run KMeans Clustering, print score and plot categorized data

int main() {

    // Specify backend renderer for matplotlib
    plt::backend("GTK3Agg");

    // create dataset with two feature
    Matrix X = read_csv("./datasets/blobs/blobs.csv");
    X = X.slice(1, X.row_length(), 0, 2);
    X.to_double();

    // plot uncategorized data
    plt::figure_size(800, 600);
    plt::title("KMeans Unclusterized Data");
    plt::plot(X.get_col(0), X.get_col(1), "ko");
    plt::save("./build/plots/KMeans Unclusterized Data.png");
    plt::show();

    // create KMeans object with k and epochs as parameters
    KMeans km(3, 100);

    Matrix Y_pred = km.fit_predict(X); // fit() and predict()
    Y_pred.to_double();
    Matrix centroid = km.get_centroid();

    std::cout << "KMeans Clustering Algorithm: " << std::endl;
    std::cout << "Score: " << km.score() << std::endl;

    // plot the 3 clusters (categorized data)
    plt::figure_size(800, 600);
    plt::title("KMeans Clusterized Data");
    plt::plot(matrix.slice_select(X, Y_pred, 0, 0).get_col(0),
              matrix.slice_select(X, Y_pred, 0, 1).get_col(0), "g1");
    plt::plot(matrix.slice_select(X, Y_pred, 1, 0).get_col(0),
              matrix.slice_select(X, Y_pred, 1, 1).get_col(0), "b^");
    plt::plot(matrix.slice_select(X, Y_pred, 2, 0).get_col(0),
              matrix.slice_select(X, Y_pred, 2, 1).get_col(0), "rD");
    plt::plot(centroid.get_col(0), centroid.get_col(1), "yx");
    plt::save("./build/plots/KMeans Clusterized Data.png");
    plt::show();

    return 0;
}
