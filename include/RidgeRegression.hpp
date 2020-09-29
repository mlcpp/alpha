#ifndef _ridge_hpp_
#define _ridge_hpp_

#include <all.hpp>
#include <preprocessing.hpp>

class Ridge {
  private:
	bool normalize, ols, is_fit = false;
	int epochs;
    double lr, alpha;
    Preprocessing preprocessing;

  public:
    Matrix B;

    Ridge(double, bool, bool, int, double);
    void fit(Matrix, Matrix);
    void get_params();
    Matrix predict(Matrix);
    double score(Matrix, Matrix);
    void set_params(double, bool, bool, int, double);
};

// Constructor
Ridge::Ridge(double alpha = 1, bool normalize = false, bool ols = false, int epochs = 100,
	     double lr = 0.1) {
    this->alpha = alpha;
    this->normalize = normalize;
    this->ols = ols;
    this->epochs = epochs;
    this->lr = lr;
}

// Method to fit the Ridge model
void Ridge::fit(Matrix X, Matrix Y) {
    if ((X.row_length() != Y.row_length()) && (X.col_length() == Y.col_length())) {
	X.T();
	Y.T();
    }

    bool expr = (X.row_length() == Y.row_length()) && (X.col_length() == Y.col_length());
    assert(("Wrong dimensions.", expr));

    if (normalize)
	X = preprocessing.normalize(X, "column");

    // Initializing parameters with zero
    B = matrix.zeros(X.col_length() + 1, 1);

    // Add a column of 1's to X
    Matrix temp_x = matrix.ones(X.row_length(), 1);
    X = matrix.concat(temp_x, X, "column");
    int m = X.row_length();
    Matrix temp;
    Matrix Y_pred;
    // ols
    if (ols) {
	Matrix L = matrix.eye(X.col_length()) * alpha;
	L(0, 0) = 0;
	L(0, 0);
	Matrix C = matrix.inverse((matrix.matmul(X.T(), X)) + L);
	Matrix D = matrix.matmul(X.T(), Y);
	B = matrix.matmul(C, D);
    }
    // gradient descent
    else {
	for (int i = 1; i <= epochs; i++) {
	    temp = matrix.concat(B.slice(0, B.row_length(), 0, 1),
				 matrix.zeros(B.row_length(), B.col_length() - 1), "column");
	    Y_pred = matrix.matmul(X, B);
	    B = B - ((matrix.matmul(X.T(), Y_pred - Y) + (B * alpha)) * (lr / m));
	    B = B + (temp * (alpha / m * lr));
	}
    }
    is_fit = true;
}

// Method to print the Ridge object parameters in json format
void Ridge::get_params() {
    std::cout << std::boolalpha;
    std::cout << "[" << std::endl;
    std::cout << "\t \"alpha\": \"" << alpha << "\"," << std::endl;
    std::cout << "\t \"normalize\": \"" << normalize << "\"," << std::endl;
    std::cout << "\t \"ols\": \"" << ols << "\"," << std::endl;
    std::cout << "\t \"epochs\": \"" << epochs << "\"," << std::endl;
    std::cout << "\t \"lr\": \"" << lr << "\"" << std::endl;
    std::cout << "]" << std::endl;
}

// Method to predict using the Ridge model
Matrix Ridge::predict(Matrix X) {
	assert(("Fit the model before predicting.", is_fit));

	if (normalize)
	X = preprocessing.normalize(X, "column");

    // Add a column of 1's to X
    Matrix temp_x = matrix.ones(X.row_length(), 1);
    X = matrix.concat(temp_x, X, "column");

    Matrix Y_pred = matrix.matmul(X, B);
    return Y_pred;
}

// Method to calculate the score of the predictions
double Ridge::score(Matrix Y_pred, Matrix Y) {
    double Y_mean = ((matrix.mean(Y, "column"))(0, 0));
    double residual_sum_of_squares = (matrix.matmul((Y_pred - Y).T(), (Y_pred - Y)))(0, 0);
    double total_sum_of_squares = (matrix.matmul((Y - Y_mean).T(), (Y - Y_mean)))(0, 0);
    double score = (1 - (residual_sum_of_squares / total_sum_of_squares));

    return score;
}

// Method to set the Ridge object parameters
void Ridge::set_params(double alpha = 1, bool normalize = false, bool ols = false, int epochs = 100,
		       double lr = 0.1) {
    this->alpha = alpha;
    this->normalize = normalize;
    this->ols = ols;
    this->epochs = epochs;
    this->lr = lr;
}

#endif /* _ridge_hpp_ */
