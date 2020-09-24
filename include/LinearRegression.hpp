#ifndef _linear_regression_hpp_
#define _linear_regression_hpp_

#include <all.hpp>

class LinearRegression {
  private:
    bool fit_intercept = true, normalize = false, copy_X = true, ols = false;
    int n_jobs = 1, epochs = 1000;
    double lr = 0.001;
    Matrix B;

  public:
    LinearRegression(bool fit_intercept = true, bool normalize = false, bool copy_X = true,
                     int n_jobs = 1, bool ols = false, int epochs = 1000, double lr = 0.001) {
        this->fit_intercept = fit_intercept;
        this->normalize = normalize;
        this->copy_X = copy_X;
        this->ols = ols;
        this->n_jobs = n_jobs;
        this->epochs = epochs;
        this->lr = lr;
    }
    Matrix fit(Matrix X, Matrix Y);
    void get_params(); // returns a map //prints json format
    Matrix predict(Matrix X);
    double score(Matrix X, Matrix Y);
    void set_params(bool fit_intercept = true, bool normalize = false, bool copy_X = true,
                    int n_jobs = 1, bool ols = false, int epochs = 1000, double lr = 0.001);
};

Matrix LinearRegression::fit(Matrix X, Matrix Y) { // estimate coefficients
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
        return B;
    }

    // gradient descent
    else {
        Matrix Y_pred;
        double learning_rate = 0.001;
        for (int i = 1; i <= epochs; i++) {
            Y_pred = matrix.matmul(X, B);
            B = B - (matrix.matmul(X.T(), Y_pred - Y)) * (lr);
        }
        return B;
    }
}

void LinearRegression::get_params() {
    std::cout << std::boolalpha;
    std::vector<std::string> params = {"fit_intercept", "normalize", "copy_X",
                                       "ols",           "n_jobs",    "epochs"};
    std::cout << "[" << std::endl;
    std::cout << "\t \"fit_intercept\": \"" << this->fit_intercept << "\"," << std::endl;
    std::cout << "\t \"normalize\": \"" << this->normalize << "\"," << std::endl;
    std::cout << "\t \"copy_X\": \"" << this->copy_X << "\"," << std::endl;
    std::cout << "\t \"ols\": \"" << this->ols << "\"," << std::endl;
    std::cout << "\t \"n_jobs\": \"" << this->n_jobs << "\"," << std::endl;
    std::cout << "\t \"epochs\": \"" << this->epochs << "\"," << std::endl;
    std::cout << "\t \"lr\": \"" << this->lr << "\"" << std::endl;
    std::cout << "]" << std::endl;
}

Matrix LinearRegression::predict(Matrix X) {
    // Add a column of 1's to X
    Matrix temp_x = matrix.ones(X.row_length(), 1);
    X = matrix.concat(temp_x, X, "column");
    Matrix Y_pred;
    Y_pred = matrix.matmul(X, B);
    return Y_pred;
}

double LinearRegression::score(Matrix Y_pred, Matrix Y) {
    double score = 0.00;
    Matrix residual_sum_of_squares = matrix.matmul((Y_pred - Y), (Y_pred - Y).T());
    Matrix total_sum_of_squares = matrix.matmul((Y - (matrix.mean(Y, "column"))(0, 0)),
                                                (Y - (matrix.mean(Y, "column"))(0, 0)).T());
    score = (1 - (residual_sum_of_squares(0, 0) / total_sum_of_squares(0, 0)));
    return score;
}

void LinearRegression::set_params(bool fit_intercept, bool normalize, bool copy_X, int n_jobs,
                                  bool ols, int epochs, double lr) {
    this->fit_intercept = fit_intercept;
    this->normalize = normalize;
    this->copy_X = copy_X;
    this->ols = ols;
    this->n_jobs = n_jobs;
    this->epochs = epochs;
    this->lr = lr;
}

#endif /* _linear_regression_hpp_ */