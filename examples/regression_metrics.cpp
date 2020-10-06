#include <LinearRegression.hpp>
#include <Matrix.hpp>
#include <metrics.hpp>

// Example program

// Read csv files to get a Matrix object.
// Slice the Matrix object to a suitable size.
// Run Linear Regression model and predict values.
// Use the Regression metrics:
//
// 1. R^2 Score
// 2. Mean squared eror regression loss
// 3. Mean absolute eror regression loss

int main() {
    // Load the dataset
    Matrix mat = read_csv("./datasets/blobs_linear/blobs_linear.csv");

    // Slice one feature for data from mat
    Matrix X = mat.slice(1, mat.row_length(), 0, 1);
    // Slice targets from mat
    Matrix y = mat.slice(1, mat.row_length(), mat.col_length() - 1, mat.col_length());

    // Convert data and targets from string to double
    X.to_double();
    y.to_double();

    // Create a linear regression object for Ordinary Least Squares
    LinearRegression regr(false, true);

    // Train the model
    regr.fit(X, y);

    // Make prediction on the dataset
    Matrix y_pred = regr.predict(X);

    // R^2 Score
    std ::cout << "R^2 Score: " << metrics.r2_score(y, y_pred) << std::endl;
    std::cout << std::endl;

    // Mean squared error regression loss
    std ::cout << "Mean Squared Error: " << metrics.mean_squared_error(y, y_pred) << std::endl;
    std::cout << std::endl;

    // Mean absolute error regression loss
    std ::cout << "Mean Absolute Error: " << metrics.mean_absolute_error(y, y_pred) << std::endl;

    return 0;
}
