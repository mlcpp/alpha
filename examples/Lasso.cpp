// Header files
#include <Lasso.hpp>
#include <Matrix.hpp>
#include <matplotlibcpp.hpp>
#include <model_selection.hpp>
namespace plt = matplotlibcpp;

/* Example program

Read csv files to get a Matrix object.
Slice the Matrix object according to our needs.
Run Lasso Regression, print scores and plot predictions in 2 ways:

1. Coordinate Descent without Normalization
2. Coordinate Descent with Normalization
*/
int main() {
    // Load the dataset
    Matrix mat = read_csv("./datasets/blobs_linear/blobs_linear.csv");
    // Slice one feature for data from mat
    Matrix X = mat.slice(1, mat.row_length(), 0, 1);

    // Slice targets from mat
    Matrix Y = mat.slice(1, mat.row_length(), mat.col_length() - 1, mat.col_length());

    // Convert data and targets from string to double
    X.to_double();
    Y.to_double();

    // Split the data and targets into training/testing sets
    auto [X_train, X_test, Y_train, Y_test] = model_selection.train_test_split(X, Y, 0);

    // Coordinate Descent without Normalization

    // Create Lasso object for Coordinate Descent without Normalization
    std::cout << "Lasso Regression using Coordinate Descent without Normalization: " << std::endl;
    Lasso regr_d;

    // Train the model using the training set
    regr_d.fit(X_train, Y_train);

    // Make prediction using the training and test set
    Matrix Y_train_pred_d = regr_d.predict(X_train);
    Matrix Y_test_pred_d = regr_d.predict(X_test);
    Matrix Y_pred_d = regr_d.predict(X);

    // The coefficient of determination = 1 is perfect prediction
    std ::cout << "Training set score: " << regr_d.score(Y_train, Y_train_pred_d) << std::endl;
    std ::cout << "Test set score: " << regr_d.score(Y_test, Y_test_pred_d) << std::endl;

    std::cout << std::endl;

    // Printing the model coefficients
    std::cout << "Model Coefficients: " << std::endl;
    regr_d.B.print();
    std::cout << std::endl << std::endl;

    plt::figure_size(800, 600);
    plt::plot(X.get_col(0), Y.get_col(0), "ro");
    plt::plot(X.get_col(0), Y_pred_d.get_col(0), "k");
    plt::title("Lasso - Coordinate Descent");
    plt::save("./examples/Lasso - Coordinate Descent.png");
    plt::show();

    // Coordinate Descent with Normalization

    // Create Lasso object for Coordinate Descent with Normalization
    std::cout << "Lasso Regression using Coordinate Descent with Normalization: " << std::endl;
    Lasso regr_n(1, true);

    // Train the model using the training set
    regr_n.fit(X_train, Y_train);

    // Make prediction using the training and test set
    Matrix Y_train_pred_n = regr_n.predict(X_train);
    Matrix Y_test_pred_n = regr_n.predict(X_test);
    Matrix Y_pred_n = regr_n.predict(X);

    // The coefficient of determination = 1 is perfect prediction
    std ::cout << "Training set score: " << regr_n.score(Y_train, Y_train_pred_n) << std::endl;
    std ::cout << "Test set score: " << regr_n.score(Y_test, Y_test_pred_n) << std::endl;

    std::cout << std::endl;

    // Printing the model coefficients
    std::cout << "Model Coefficients: " << std::endl;
    regr_n.B.print();

    plt::figure_size(800, 600);
    plt::plot(X.get_col(0), Y.get_col(0), "ro");
    plt::plot(X.get_col(0), Y_pred_n.get_col(0), "k");
    plt::title("Lasso - Coordinate Descent with Normalization");
    plt::save("./examples/Lasso - Coordinate Descent with Normalization.png");
    plt::show();

    return 0;
}