// Header files
#include <Lasso.hpp>
#include <Matrix.hpp>
#include <matplotlibcpp.hpp>
#include <model_selection.hpp>
namespace plt = matplotlibcpp;

// Example program

// Read csv files to get a Matrix object.
// Slice the Matrix object according to our needs.
// Run Lasso Regression, print scores and plot predictions in 2 ways:

// 1. Coordinate Descent without Normalization
// 2. Coordinate Descent with Normalization

int main() {
    // Specify backend renderer for matplotlib
    plt::backend("GTK3Agg");

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
    Lasso regr_cd;

    // Train the model using the training set
    regr_cd.fit(X_train, Y_train);

    // Make prediction using the training and test set
    Matrix Y_train_pred_cd = regr_cd.predict(X_train);
    Matrix Y_test_pred_cd = regr_cd.predict(X_test);
    Matrix Y_pred_cd = regr_cd.predict(X);

    // The coefficient of determination = 1 is perfect prediction
    std ::cout << "Training set score: " << regr_cd.score(Y_train, Y_train_pred_cd) << std::endl;
    std ::cout << "Test set score: " << regr_cd.score(Y_test, Y_test_pred_cd) << std::endl;

    std::cout << std::endl;

    // Printing the model coefficients
    std::cout << "Model Coefficients: " << std::endl;
    regr_cd.B.print();
    std::cout << std::endl << std::endl;

    plt::figure_size(800, 600);
    plt::plot(X.get_col(0), Y.get_col(0), "ro");
    plt::plot(X.get_col(0), Y_pred_cd.get_col(0), "k");
    plt::title("Lasso - Coordinate Descent");
    plt::save("./build/plots/Lasso - Coordinate Descent.png");
    plt::show();

    // Coordinate Descent with Normalization

    // Create Lasso object for Coordinate Descent with Normalization
    std::cout << "Lasso Regression using Coordinate Descent with Normalization: " << std::endl;
    Lasso regr_cdn(1, true);

    // Train the model using the training set
    regr_cdn.fit(X_train, Y_train);

    // Make prediction using the training and test set
    Matrix Y_train_pred_cdn = regr_cdn.predict(X_train);
    Matrix Y_test_pred_cdn = regr_cdn.predict(X_test);
    Matrix Y_pred_cdn = regr_cdn.predict(X);

    // The coefficient of determination = 1 is perfect prediction
    std ::cout << "Training set score: " << regr_cdn.score(Y_train, Y_train_pred_cdn) << std::endl;
    std ::cout << "Test set score: " << regr_cdn.score(Y_test, Y_test_pred_cdn) << std::endl;

    std::cout << std::endl;

    // Printing the model coefficients
    std::cout << "Model Coefficients: " << std::endl;
    regr_cdn.B.print();

    plt::figure_size(800, 600);
    plt::plot(X.get_col(0), Y.get_col(0), "g^");
    plt::plot(X.get_col(0), Y_pred_cdn.get_col(0), "k");
    plt::title("Lasso - Coordinate Descent with Normalization");
    plt::save("./build/plots/Lasso - Coordinate Descent with Normalization.png");
    plt::show();

    return 0;
}
