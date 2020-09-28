// import header files
#include <KMeans.hpp>
#include <matplotlibcpp.hpp>
namespace plt = matplotlibcpp;

int main() {
    // create dataset with two feature
    Matrix X = read_csv("./datasets/blobs/blobs.csv");
    X = X.slice(1, X.row_length(), 0, 2);
    X.to_double();
    // plot uncategorized data
    plt::plot(X.get_col(0), X.get_col(1), "ko");
    plt::show();

    // create KMeans object
    KMeans km(3, 100);

    Matrix Y_pred = km.fit_predict(X); // fit() and predict()
    Y_pred.to_double();
    Matrix centroid = km.get_centroid();

    // plot the 3 clusters (categorized data)
    plt::plot(matrix.slice_select(X, Y_pred, 0, 0).get_col(0),
              matrix.slice_select(X, Y_pred, 0, 1).get_col(0), "g1");
    plt::plot(matrix.slice_select(X, Y_pred, 1, 0).get_col(0),
              matrix.slice_select(X, Y_pred, 1, 1).get_col(0), "b^");
    plt::plot(matrix.slice_select(X, Y_pred, 2, 0).get_col(0),
              matrix.slice_select(X, Y_pred, 2, 1).get_col(0), "rD");
    plt::plot(centroid.get_col(0), centroid.get_col(1), "yx");

    plt::show();
}