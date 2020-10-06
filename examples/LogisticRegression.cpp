// Include necessary header files
#include <LogisticRegression.hpp>
#include <Matrix.hpp>
#include <matplotlibcpp.hpp>
#include <model_selection.hpp>
namespace plt = matplotlibcpp;

// Example program

// Read csv files to get a Matrix object.
// Slice the Matrix object to a suitable size.
// Run Logistic Regression, print scores and plot predictions in 2 ways:

// 1. Gradient Descent without Regularization
// 2. Gradient Descent with Regularization

int main() {
    // Specify backend renderer for matplotlib
    plt::backend("GTK3Agg");
    // Load the dataset
    Matrix mat = read_csv("./datasets/blobs_clf/blobs_clf.csv");

    // Slice one feature for data from mat
    Matrix X = mat.slice(1, mat.row_length(), 0, mat.col_length() - 1);
    // Slice targets from mat
    Matrix Y = mat.slice(1, mat.row_length(), mat.col_length() - 1, mat.col_length());

    // Convert data and targets from string to double
    X.to_double();
    Y.to_double();

    // Split the data and targets into training/testing sets
    auto [X_train, X_test, Y_train, Y_test] = model_selection.train_test_split(X, Y, 0);

    // Without Regularization

    // Create Logistic regression object for Gradient Descent without Regularization
    std::cout << "Logistic Regression without Regularization: " << std::endl;
    LogisticRegression clf("none");

    // Train the model using the training set
    clf.fit(X_train, Y_train);

    // Make prediction using the training and test set
    Matrix Y_train_pred = clf.predict(X_train);
    Matrix Y_test_pred = clf.predict(X_test);
    Matrix Y_pred = clf.predict(X);

    // The coefficient of determination = 1 is perfect prediction
    std ::cout << "Training set score: " << clf.score(Y_train, Y_train_pred) << std::endl;
    std ::cout << "Test set score: " << clf.score(Y_test, Y_test_pred) << std::endl;

    std::cout << std::endl;

    // Printing the model coefficients
    std::cout << "Model Coefficients: " << std::endl;
    clf.B.print();
    std::cout << std::endl << std::endl;

    // Y_pred.print();

    plt::figure_size(800, 600);
    plt::named_plot("Class 0", matrix.slice_select(X, Y, 0, 0).get_col(0),
                    matrix.slice_select(X, Y, 0, 1).get_col(0), "ro");
    plt::named_plot("Class 1", matrix.slice_select(X, Y, 1, 0).get_col(0),
                    matrix.slice_select(X, Y, 1, 1).get_col(0), "g^");
    plt::title("Logistic Regression - Known Dataset");
    plt::legend();
    plt::show();

    plt::figure_size(800, 600);
    plt::named_plot("Predicted: Class 0", matrix.slice_select(X, Y_pred, 0, 0.0).get_col(0),
                    matrix.slice_select(X, Y_pred, 0, 1.0).get_col(0), "mo");
    plt::named_plot("Predicted: Class 1", matrix.slice_select(X, Y_pred, 1, 0.0).get_col(0),
                    matrix.slice_select(X, Y_pred, 1, 1.0).get_col(0), "k^");
    plt::title("Logistic Regression without Regularization - Predictions");
    plt::legend();
    plt::save("./build/plots/LogisticRegression without Regularization.png");
    plt::show();

    // With Regularization

    // Create Logistic regression object for Gradient Descent with Regularization
    std::cout << "Logistic Regression with Regularization: " << std::endl;
    LogisticRegression clf_reg;

    // Train the model using the training set
    clf_reg.fit(X_train, Y_train);

    // Make prediction using the training and test set
    Matrix Y_train_pred_reg = clf_reg.predict(X_train);
    Matrix Y_test_pred_reg = clf_reg.predict(X_test);
    Matrix Y_pred_reg = clf_reg.predict(X);

    // The coefficient of determination = 1 is perfect prediction
    std ::cout << "Training set score: " << clf_reg.score(Y_train, Y_train_pred_reg) << std::endl;
    std ::cout << "Test set score: " << clf_reg.score(Y_test, Y_test_pred_reg) << std::endl;

    std::cout << std::endl;

    // Printing the model coefficients
    std::cout << "Model Coefficients: " << std::endl;
    clf_reg.B.print();

    plt::figure_size(800, 600);
    plt::named_plot("Class 0", matrix.slice_select(X, Y, 0, 0).get_col(0),
                    matrix.slice_select(X, Y, 0, 1).get_col(0), "ro");
    plt::named_plot("Class 1", matrix.slice_select(X, Y, 1, 0).get_col(0),
                    matrix.slice_select(X, Y, 1, 1).get_col(0), "g^");
    plt::title("Logistic Regression - Known Dataset");
    plt::legend();
    plt::show();

    plt::figure_size(800, 600);
    plt::named_plot("Predicted: Class 0", matrix.slice_select(X, Y_pred, 0, 0.0).get_col(0),
                    matrix.slice_select(X, Y_pred, 0, 1.0).get_col(0), "mo");
    plt::named_plot("Predicted: Class 1", matrix.slice_select(X, Y_pred, 1, 0.0).get_col(0),
                    matrix.slice_select(X, Y_pred, 1, 1.0).get_col(0), "k^");
    plt::title("Logistic Regression with Regularization - Predictions");
    plt::legend();
    plt::save("./build/plots/LogisticRegression with Regularization.png");
    plt::show();

    return 0;
}
