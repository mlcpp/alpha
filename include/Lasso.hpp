#ifndef _lasso_hpp_
#define _lasso_hpp_

#include <all.hpp>
#include <preprocessing.hpp>

class Lasso : private Preprocessing {
  private:
    bool to_normalize, is_fit = false;
    int epochs;
    double alpha;

    double S(double);

  public:
    Matrix B;

    Lasso(double, bool, int);
    void fit(Matrix, Matrix);
    void get_params();
    void path();
    Matrix predict(Matrix);
    double score(Matrix, Matrix);
    void set_params(double, bool, int);
};

// Constructor
Lasso::Lasso(double alpha = 1, bool normalize = false, int epochs = 100) {
    this->alpha = alpha;
    to_normalize = normalize;
    this->epochs = epochs;
}

// Method to fit the Lasso model
void Lasso::fit(Matrix X, Matrix Y) {
    if ((X.row_length() != Y.row_length()) && (X.col_length() == Y.col_length())) {
        X.T();
        Y.T();
    }

    bool expr = (X.row_length() == Y.row_length()) || (X.col_length() == Y.col_length());
    assert(("Wrong dimensions.", expr));

    if (to_normalize)
        X = normalize(X, "column");

    // Initializing parameters with zero
    B = matrix.zeros(X.col_length() + 1, 1);

    // Add a column of 1's to X
    Matrix temp_x = matrix.ones(X.row_length(), 1);
    X = matrix.concat(temp_x, X, "column");
    int m = X.row_length();
    Matrix temp;
    Matrix Y_pred;

    // coordinate descent
    for (int i = 1; i <= epochs; i++) {
        for (int j = 0; j < X.col_length(); j++) {
            double p = ((matrix.matmul(
                (Y - (matrix.matmul(matrix.del(X, j, "column"), matrix.del(B, j, "row")))).T(),
                X.slice(0, X.row_length(), j, j + 1))))(0, 0);
            Matrix temp = X.slice(0, X.row_length(), j, j + 1);
            double z = (matrix.matmul(temp.T(), temp))(0, 0);
            B(j, 0) = S(p) / z;
        }
    }
    is_fit = true;
}

// Method to print the Lasso object parameters in json format
void Lasso::get_params() {
    std::cout << std::boolalpha;
    std::cout << "[" << std::endl;
    std::cout << "\t \"alpha\": \"" << alpha << "\"," << std::endl;
    std::cout << "\t \"normalize\": \"" << to_normalize << "\"," << std::endl;
    std::cout << "\t \"epochs\": \"" << epochs << "\"," << std::endl;
    std::cout << "]" << std::endl;
}

// Method to predict using the Lasso model
Matrix Lasso::predict(Matrix X) {
    assert(("Fit the model before predicting.", is_fit));

    if (to_normalize)
        X = normalize(X, "column");

    // Add a column of 1's to X
    Matrix temp_x = matrix.ones(X.row_length(), 1);
    X = matrix.concat(temp_x, X, "column");

    Matrix Y_pred = matrix.matmul(X, B);
    return Y_pred;
}

// Method to calculate the score of the predictions
double Lasso::score(Matrix Y_pred, Matrix Y) {
    double Y_mean = ((matrix.mean(Y, "column"))(0, 0));
    double residual_sum_of_squares = (matrix.matmul((Y_pred - Y).T(), (Y_pred - Y)))(0, 0);
    double total_sum_of_squares = (matrix.matmul((Y - Y_mean).T(), (Y - Y_mean)))(0, 0);
    double score = (1 - (residual_sum_of_squares / total_sum_of_squares));

    return score;
}

// Method to set the Lasso object parameters
void Lasso::set_params(double alpha = 1, bool normalize = false, int epochs = 100) {
    this->alpha = alpha;
    to_normalize = normalize;
    this->epochs = epochs;
}

// Helper methods

// Method to return soft thresholding
double Lasso::S(double p) {
    if (p < -alpha)
        return p + alpha;
    else if (p > alpha)
        return p - alpha;
    else
        return 0;
}

#endif /* _lasso_hpp_ */
