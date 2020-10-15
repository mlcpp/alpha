// import header files
#include <KNeighborsClassifier.hpp>
#include <matplotlibcpp.hpp>
#include <model_selection.hpp>
namespace plt = matplotlibcpp;

// Example program

// Read csv files to get a Matrix object.
// Slice the Matrix object to a suitable size.
// Plot known data points
// Run KNeighborsClustering and predict the cluster label for a Matrix
// Print score and plot necessary graphs

int main() {

    // Specify backend renderer for matplotlib
    plt::backend("GTK3Agg");

    // create dataset with two feature
    Matrix mat = read_csv("./datasets/blobs/blobs.csv");
    Matrix X = mat.slice(1, mat.row_length(), 0, 2);
    Matrix Y = mat.slice(1, mat.row_length(), 2, 3);
    X.to_double();
    Y.to_double();

    // Split the data and targets into training/testing sets
    auto [X_train, X_test, Y_train, Y_test] = model_selection.train_test_split(X, Y, 0);

    // plot training dataset
    plt::figure_size(800, 600);
    plt::title("KNeighbors Known Dataset");
    plt::plot(matrix.slice_select(X_train, Y_train, 0.0, 0).get_col(0),
              matrix.slice_select(X_train, Y_train, 0.0, 1).get_col(0), "ro");
    plt::plot(matrix.slice_select(X_train, Y_train, 1.0, 0).get_col(0),
              matrix.slice_select(X_train, Y_train, 1.0, 1).get_col(0), "g^");
    plt::plot(matrix.slice_select(X_train, Y_train, 2.0, 0).get_col(0),
              matrix.slice_select(X_train, Y_train, 2.0, 1).get_col(0), "bD");
    plt::save("./build/plots/KNeighbors Known Dataset.png");
    plt::show();

    // plot test data
    plt::figure_size(800, 600);
    plt::title("KNeighbors Unknown Dataset");
    plt::plot(X_test.get_col(0), X_test.get_col(1), "mo");
    plt::save("./build/plots/KNeighbors Unknown Dataset.png");
    plt::show();

    // create KMeans object with k and epochs as parameters
    KNeighborsClassifier knn(1);
    knn.fit(X_train, Y_train);

    std::cout << "K Neighbors Clustering Algorithm: " << std::endl;
    Matrix Y_pred = knn.predict(X_test);

    // plot predicted dataset
    plt::figure_size(800, 600);
    plt::title("KNeighbors Predicted Dataset along with Known Dataset");
    plt::named_plot("Known red", matrix.slice_select(X_train, Y_train, 0.0, 0).get_col(0),
                    matrix.slice_select(X_train, Y_train, 0.0, 1).get_col(0), "ro");
    plt::named_plot("Known green", matrix.slice_select(X_train, Y_train, 1.0, 0).get_col(0),
                    matrix.slice_select(X_train, Y_train, 1.0, 1).get_col(0), "g^");
    plt::named_plot("Known blue", matrix.slice_select(X_train, Y_train, 2.0, 0).get_col(0),
                    matrix.slice_select(X_train, Y_train, 2.0, 1).get_col(0), "bD");

    plt::named_plot("Predicted red", matrix.slice_select(X_test, Y_pred, 0.0, 0).get_col(0),
                    matrix.slice_select(X_test, Y_pred, 0.0, 1).get_col(0), "rP");
    plt::named_plot("Predicted green", matrix.slice_select(X_test, Y_pred, 1.0, 0).get_col(0),
                    matrix.slice_select(X_test, Y_pred, 1.0, 1).get_col(0), "gP");
    plt::named_plot("Predicted blue", matrix.slice_select(X_test, Y_pred, 2.0, 0).get_col(0),
                    matrix.slice_select(X_test, Y_pred, 2.0, 1).get_col(0), "bP");
    plt::legend();
    plt::save("./build/plots/KNeighbors Predicted Dataset.png");
    plt::show();

    // Comparison of predicted and actual cluster label
    std::cout << "KNN Model Score: " << knn.score(Y_pred, Y_test) << std::endl;
    return 0;
}
