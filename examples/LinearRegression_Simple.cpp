// Header files
#include <LinearRegression.hpp>
#include <Matrix.hpp>
#include <matplotlibcpp.hpp>
#include <model_selection.hpp>
namespace plt = matplotlibcpp;

int main() {

    // Load the datasets
    Matrix mat = read_csv("./datasets/diabetes/diabetes.csv");

    // Slice one feature for data from mat
    Matrix X = mat.slice(1, mat.row_length(), 0, 1);

    // Slice targets from mat
    Matrix Y = mat.slice(1, mat.row_length(), mat.col_length() - 1, mat.col_length());

    // Convert data and targets from string to double
    X.to_double();
    Y.to_double();

    // Split the data and targets into training/testing sets
    auto [X_train, X_test, Y_train, Y_test] = model_selection.train_test_split(X, Y, 0, 0.95, 0.05);

    // Create linear regression object
    LinearRegression regr;

    // Train the model using the training set
    regr.fit(X_train, Y_train);

    // Make prediction using the testing set
    Matrix Y_train_pred = regr.predict(X_train);
    Matrix Y_pred = regr.predict(X_test);

    // The mean squared error
    // std :: cout << mean_squared_error(diabetes_y_test, diabetes_y_pred);

    // The coefficient of determination: 1 is perfect prediction
    std ::cout << regr.score(Y_train, Y_train_pred) << std::endl;
    std ::cout << regr.score(Y_test, Y_pred) << std::endl;

    plt::figure_size(800, 600);
    plt::named_plot("Age", X_test.get_col(0), Y_test.get_col(0), "ro");
    plt::named_plot("Predicted Age", X_test.get_col(0), Y_pred.get_col(0), "k");
    plt::title("Simple Linear Regression");
    plt::save("./examples/LinearRegression_Simple.png");
    plt::legend();
    plt::show();
    return 0;
}