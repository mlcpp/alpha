// Header files
#include <Matrix.hpp>
#include <Ridge.hpp>
#include <matplotlibcpp.hpp>
#include <model_selection.hpp>
namespace plt = matplotlibcpp;

int main() {

    // Load the dataset
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
    Ridge regr(1, true);

    // Train the model using the training set
    regr.fit(X_train, Y_train);

    // Make prediction using the training and test set
    Matrix Y_train_pred = regr.predict(X_train);
    Matrix Y_test_pred = regr.predict(X_test);

    // The coefficient of determination = 1 is perfect prediction
    std ::cout << "Training set score: " << regr.score(Y_train, Y_train_pred) << std::endl;
    std ::cout << "Test set score: " << regr.score(Y_test, Y_test_pred) << std::endl;

    std::cout << std::endl;

    // Printing the model coefficients
    std::cout << "Model Coefficients: " << std::endl;
    regr.B.print();

    plt::figure_size(800, 600);
    plt::named_plot("Age", X_test.get_col(0), Y_test.get_col(0), "ro");
    plt::named_plot("Predicted Age", X_test.get_col(0), Y_test_pred.get_col(0), "k");
    plt::title("Simple Linear Regression");
    plt::save("./examples/LinearRegression_Simple.png");
    plt::legend();
    plt::show();

    return 0;
}