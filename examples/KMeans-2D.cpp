// import header files
#include <KMeans.hpp>
#include <matplotlibcpp.hpp>
namespace plt = matplotlibcpp;

int main() {
    // create dataset with two feature
    Matrix X = read_csv("./datasets/make_blob/make_blob.csv");
    X.to_double();
    // plot uncategorized data
    plt::plot(X.slice(0, X.row_length(), 0, 1), X.slice(0, X.row_length(), 0, 1), "ro");
    plt::show();

    // create KMeans object
    KMeans km(3, 1000);
    Matrix centroid = km.get_centroid();
    Matrix Y_pred = km.fit_predict(X); // fit() and predict()

    // plot the 3 clusters (categorized data)
    // (first feature where y = 0, second feature)
    plt::plot(X.slice_select(Y_pred, 0, 0), X.slice_select(Y_pred, 0, 1), "g^");
    plt::plot(X.slice_select(Y_pred, 1, 0), X.slice_select(Y_pred, 0, 1), "k^");
    plt::plot(X.slice_select(Y_pred, 2, 0), X.slice_select(Y_pred, 0, 1), "y^");
    plt::plot(centroid.get_col(0), centroid.get_col(1), "*");
    plt::show();
}