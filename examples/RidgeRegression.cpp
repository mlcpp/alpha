// Header files
#include <Matrix.hpp>
#include <Ridge.hpp>
#include <matplotlibcpp.hpp>
#include <model_selection.hpp>
namespace plt = matplotlibcpp;

/* Example program

Read csv files to get a Matrix object.
Slice the Matrix object according to our needs.
Run Ridge Regression, print scores and plot predictions in 3 ways:

1. Gradient Descent without Normalization
2. Gradient Descent with Normalization
3. Ordinary Least Squares
*/
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

    // Gradient Descent without Normalization

    // Create Ridge regression object for Gradient Descent without Normalization
    std::cout << "Ridge Regression using Gradient Descent without Normalization: " << std::endl;
    Ridge regr_gd;

    // Train the model using the training set
    regr_gd.fit(X_train, Y_train);

    // Make prediction using the training and test set
    Matrix Y_train_pred_gd = regr_gd.predict(X_train);
    Matrix Y_test_pred_gd = regr_gd.predict(X_test);
    Matrix Y_pred_gd = regr_gd.predict(X);

    // The coefficient of determination = 1 is perfect prediction
    std ::cout << "Training set score: " << regr_gd.score(Y_train, Y_train_pred_gd) << std::endl;
    std ::cout << "Test set score: " << regr_gd.score(Y_test, Y_test_pred_gd) << std::endl;

    std::cout << std::endl;

    // Printing the model coefficients
    std::cout << "Model Coefficients: " << std::endl;
    regr_gd.B.print();
    std::cout << std::endl << std::endl;

    plt::figure_size(800, 600);
    plt::plot(X.get_col(0), Y.get_col(0), "ro");
    plt::plot(X.get_col(0), Y_pred_gd.get_col(0), "k");
    plt::title("Ridge - Gradient Descent");
    plt::save("./build/plots/Ridge - Gradient Descent.png");
    plt::show();

    // Gradient Descent with Normalization

    // Create Ridge regression object for Gradient Descent with Normalization
    std::cout << "Ridge Regression using Gradient Descent with Normalization: " << std::endl;
    Ridge regr_gdn(1, true);

    // Train the model using the training set
    regr_gdn.fit(X_train, Y_train);

    // Make prediction using the training and test set
    Matrix Y_train_pred_gdn = regr_gdn.predict(X_train);
    Matrix Y_test_pred_gdn = regr_gdn.predict(X_test);
    Matrix Y_pred_gdn = regr_gdn.predict(X);

    // The coefficient of determination = 1 is perfect prediction
    std ::cout << "Training set score: " << regr_gdn.score(Y_train, Y_train_pred_gdn) << std::endl;
    std ::cout << "Test set score: " << regr_gdn.score(Y_test, Y_test_pred_gdn) << std::endl;

    std::cout << std::endl;

    // Printing the model coefficients
    std::cout << "Model Coefficients: " << std::endl;
    regr_gdn.B.print();
    std::cout << std::endl << std::endl;

    plt::figure_size(800, 600);
    plt::plot(X.get_col(0), Y.get_col(0), "g^");
    plt::plot(X.get_col(0), Y_pred_gdn.get_col(0), "k");
    plt::title("Ridge - Gradient Descent with Normalization");
    plt::save("./build/plots/Ridge - Gradient Descent with Normalization.png");
    plt::show();

    // Ordinary Least Squares

    // Create Ridge regression object for Ordinary Least Squares
    std::cout << "Ridge Regression using Ordinary Least Squares: " << std::endl;
    Ridge regr_ols(1, false, true);

    // Train the model using the training set
    regr_ols.fit(X_train, Y_train);

    // Make prediction using the training and test set
    Matrix Y_train_pred_ols = regr_ols.predict(X_train);
    Matrix Y_test_pred_ols = regr_ols.predict(X_test);
    Matrix Y_pred_ols = regr_ols.predict(X);

    // The coefficient of determination = 1 is perfect prediction
    std ::cout << "Training set score: " << regr_ols.score(Y_train, Y_train_pred_ols) << std::endl;
    std ::cout << "Test set score: " << regr_ols.score(Y_test, Y_test_pred_ols) << std::endl;

    std::cout << std::endl;

    // Printing the model coefficients
    std::cout << "Model Coefficients: " << std::endl;
    regr_ols.B.print();

    plt::figure_size(800, 600);
    plt::plot(X.get_col(0), Y.get_col(0), "bD");
    plt::plot(X.get_col(0), Y_pred_ols.get_col(0), "k");
    plt::title("Ridge - Ordinary Least Squares");
    plt::save("./build/plots/Ridge - Ordinary Least Squares.png");
    plt::show();

    return 0;
}
