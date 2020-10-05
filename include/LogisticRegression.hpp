#ifndef _logistic_regression_hpp_
#define _logistic_regression_hpp_

#include <all.hpp>

class LogisticRegression {
  private:
    std::string penalty;
    bool is_fit = false;
    int epochs;
    double lr, C;

    Matrix sigmoid(Matrix);

  public:
    Matrix B;

    LogisticRegression(std::string, int, double, double);
    void fit(Matrix, Matrix);
    void get_params();
    Matrix predict(Matrix);
    Matrix predict_log_proba(Matrix);
    Matrix predict_proba(Matrix);
    double score(Matrix, Matrix);
    void set_params(std::string, int, double, double);
};

// Constructor
LogisticRegression::LogisticRegression(std::string penalty = "l2", int epochs = 100,
                                       double lr = 0.1, double C = 1) {
    this->penalty = penalty;
    this->epochs = epochs;
    this->lr = lr;
    this->C = C;
}

// Method to fit the Logistic Regression model
void LogisticRegression::fit(Matrix X, Matrix Y) {
    if ((X.row_length() != Y.row_length()) && (X.col_length() == Y.col_length())) {
        X.T();
        Y.T();
    }

    bool expr = (X.row_length() == Y.row_length()) || (X.col_length() == Y.col_length());
    assert(("Wrong dimensions.", expr));

    // Initializing parameters with zero
    B = matrix.zeros(X.col_length() + 1, 1);

    // Add a column of 1's to X
    Matrix temp_x = matrix.ones(X.row_length(), 1);
    X = matrix.concat(temp_x, X, "column");

    // Withour Regularization
    if (penalty == "none") {
        Matrix Y_pred;
        int m = X.row_length();
        for (int i = 1; i <= epochs; i++) {
            Y_pred = sigmoid(matrix.matmul(X, B));
            B = B - ((matrix.matmul(X.T(), Y_pred - Y)) * (lr / m));
        }
    }
    // L2 Regularization
    else if (penalty == "l2") {
        Matrix Y_pred;
        int m = X.row_length();
        for (int i = 1; i <= epochs; i++) {
            Y_pred = sigmoid(matrix.matmul(X, B));
            B = B - (((matrix.matmul(X.T(), Y_pred - Y) * C) + (B)) * (lr / m));
        }
    } else {
        assert(
            ("Value of 'penalty' parameter is wrong. Please use set_params() to change.", false));
    }
    is_fit = true;
}

// Method to print the Logistic Regression object parameters in json format
void LogisticRegression::get_params() {
    std::cout << std::boolalpha;
    std::cout << "[" << std::endl;
    std::cout << "\t \"penalty\": \"" << penalty << "\"," << std::endl;
    std::cout << "\t \"epochs\": \"" << epochs << "\"," << std::endl;
    std::cout << "\t \"lr\": \"" << lr << "\"," << std::endl;
    std::cout << "\t \"C\": \"" << C << "\"" << std::endl;
    std::cout << "]" << std::endl;
}

// Method to predict class labels using the Logistic Regression model
Matrix LogisticRegression::predict(Matrix X) {
    assert(("Fit the model before predicting.", is_fit));

    // Add a column of 1's to X
    Matrix temp_x = matrix.ones(X.row_length(), 1);
    X = matrix.concat(temp_x, X, "column");

    Matrix Y_pred = matrix.matmul(X, B);
    for (int i = 0; i < Y_pred.row_length(); i++) {
        if (Y_pred(i, 0) >= 0)
            Y_pred(i, 0) = 1;
        else
            Y_pred(i, 0) = 0;
    }
    Y_pred(Y_pred.row_length() - 1, 0);
    return Y_pred;
}

// Method to predict logarithm of probablility estimates using the Logistic Regression model
Matrix LogisticRegression::predict_log_proba(Matrix X) {
    assert(("Fit the model before predicting.", is_fit));

    // Add a column of 1's to X
    Matrix temp_x = matrix.ones(X.row_length(), 1);
    X = matrix.concat(temp_x, X, "column");

    Matrix Y_pred = matrix.log(sigmoid(matrix.matmul(X, B)));
    return Y_pred;
}

// Method to predict probablility estimates using the Logistic Regression model
Matrix LogisticRegression::predict_proba(Matrix X) {
    assert(("Fit the model before predicting.", is_fit));

    // Add a column of 1's to X
    Matrix temp_x = matrix.ones(X.row_length(), 1);
    X = matrix.concat(temp_x, X, "column");

    Matrix Y_pred = sigmoid(matrix.matmul(X, B));
    return Y_pred;
}

// Method to calculate the score of the predictions
double LogisticRegression::score(Matrix Y_pred, Matrix Y) {
    double count = 0;
    for (int i = 0; i < Y.row_length(); i++) {
        if (Y(i, 0) == Y_pred(i, 0)) {
            count++;
        }
    }

    return count / Y.row_length();
}

// Method to set the Logistic Regression object parameters
void LogisticRegression::set_params(std::string penalty = "l2", int epochs = 100, double lr = 0.1,
                                    double C = 1) {
    this->penalty = penalty;
    this->epochs = epochs;
    this->lr = lr;
    this->C = C;
}

// Helper methods

Matrix LogisticRegression::sigmoid(Matrix mat) {
    mat = matrix.exp(-mat);
    mat = mat + 1;
    mat = matrix.reciprocal(mat);
    return mat;
}

#endif /* _logistic_regression_hpp_ */
