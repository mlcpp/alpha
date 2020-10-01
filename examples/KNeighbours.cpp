// import header files
#include <KNeighborsClassifier.hpp>
#include <matplotlibcpp.hpp>
#include <model_selection.hpp>
namespace plt = matplotlibcpp;

// Example program

// Read csv files to get a Matrix object.
// Slice the Matrix object to a suitable size.
// Plot known data points
// Run KNeighborsClustering and predict the cluster label for a point and a Matrix
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
    plt::plot(matrix.slice_select(X_train, Y_train, 0, 0).get_col(0), matrix.slice_select(X_train, Y_train, 0, 1).get_col(0), "ro");
    plt::plot(matrix.slice_select(X_train, Y_train, 1, 0).get_col(0), matrix.slice_select(X_train, Y_train, 1, 1).get_col(0), "g^");
    plt::plot(matrix.slice_select(X_train, Y_train, 2, 0).get_col(0), matrix.slice_select(X_train, Y_train, 2, 1).get_col(0), "bD");
    plt::save("./build/plots/KNeighbors Known Dataset.png");
    plt::show();

    // // create KMeans object with k and epochs as parameters
    // KNeighborsClustering knn(3, 100);
    // knn.fit(X_train, Y_train);

    // std::cout << "KNeighborsClustering Algorithm: " << std::endl;
    // // Identifying the cluster label for a point P = (x, y)
    // auto label = knn.transform(0, 2);
    // std::cout << label;

    // // Identifying the cluster label for a Matrix of test data
    // Matrix Y_pred = knn.predict(X_test);
    // Y_pred.to_double();
    // Y_pred.print();

    // std::cout << "Score: " << km.score() << std::endl;

    // // Comparison of predicted and actual cluster labels
    // knn.compare(Y_test, Y_label);

    return 0;
}
