#ifndef _linear_regression_hpp_
#define _linear_regression_hpp_

#include <all.hpp>

class LinearRegression {
  private:
    bool normalize, ols, if_fit = false;
    int epochs;
    double lr;

  public:
    Matrix B;

    LinearRegression(bool, bool, int, double);
    void fit(Matrix, Matrix);
    void get_params();
    Matrix predict(Matrix);
    double score(Matrix, Matrix);
    void set_params(bool, bool, int, double);
};

// Constructor
LinearRegression::LinearRegression(bool normalize = false, bool ols = false, int epochs = 1000,
                                   double lr = 0.001) {
    this->normalize = normalize;
    this->ols = ols;
    this->epochs = epochs;
    this->lr = lr;
}

// Method to fit the Linear Regression model
void LinearRegression::fit(Matrix X, Matrix Y) {
    if ((X.row_length() != Y.row_length()) && (X.col_length() == Y.col_length())) {
        X.T();
        Y.T();
    }
    bool expr = (X.row_length() == Y.row_length()) && (X.col_length() == Y.col_length());
    assert(("Wrong dimensions.", expr));

    // Initializing parameters with zero
    B = matrix.zeros(X.col_length() + 1, 1);

    // Add a column of 1's to X
    Matrix temp_x = matrix.ones(X.row_length(), 1);
    X = matrix.concat(temp_x, X, "column");

    // ols
    if (ols) {
        Matrix C = matrix.inverse((matrix.matmul(X.T(), X)));
        Matrix D = matrix.matmul(X.T(), Y);
        B = matrix.matmul(C, D);
    }
    // gradient descent
    else {
        Matrix Y_pred;
        int m = X.row_length();
        for (int i = 1; i <= epochs; i++) {
            Y_pred = matrix.matmul(X, B);
            B = B - (matrix.matmul(X.T(), Y_pred - Y)) * (lr / m);
        }
    }
    if_fit = true;
}

// Method to print the Linear Regression object parameters in json format
void LinearRegression::get_params() {
    std::cout << std::boolalpha;
    std::cout << "[" << std::endl;
    std::cout << "\t \"normalize\": \"" << normalize << "\"," << std::endl;
    std::cout << "\t \"ols\": \"" << ols << "\"," << std::endl;
    std::cout << "\t \"epochs\": \"" << epochs << "\"," << std::endl;
    std::cout << "\t \"lr\": \"" << lr << "\"" << std::endl;
    std::cout << "]" << std::endl;
}

// Method to predict using the Linear Regression model
Matrix LinearRegression::predict(Matrix X) {
    assert(("Fit the model before predicting.", if_fit));

    // Add a column of 1's to X
    Matrix temp_x = matrix.ones(X.row_length(), 1);
    X = matrix.concat(temp_x, X, "column");
    Matrix Y_pred;
    Y_pred = matrix.matmul(X, B);
    return Y_pred;
}

// Method to calculate the score of the predictions
double LinearRegression::score(Matrix Y_pred, Matrix Y) {
    double Y_mean = ((matrix.mean(Y, "column"))(0, 0));
    double residual_sum_of_squares = (matrix.matmul((Y_pred - Y).T(), (Y_pred - Y)))(0, 0);
    double total_sum_of_squares = (matrix.matmul((Y - Y_mean).T(), (Y - Y_mean)))(0, 0);
    double score = (1 - (residual_sum_of_squares / total_sum_of_squares));

    return score;
}

// Method to set the Linear Regression object parameters
void LinearRegression::set_params(bool normalize = false, bool ols = false, int epochs = 1000,
                                  double lr = 0.001) {
    this->normalize = normalize;
    this->ols = ols;
    this->epochs = epochs;
    this->lr = lr;
}

#endif /* _linear_regression_hpp_ */