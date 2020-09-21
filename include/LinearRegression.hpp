#ifndef _linear_regression_hpp_
#define _linear_regression_hpp_

#include <all.h>

template <typename T>
class LinearRegression{
	private:
		bool fit_intercept = true, normalize=false,  copy_X = true, ols = false;
		int n_jobs = 1, epochs = 1000;
		Matrix B, B_dash;
	public:
	void fit(Matrix X, Matrix Y);
	void get_params(); // returns a map //prints json format
	void predict(Matrix X);
	void score(Matrix X, Matrix Y);
	void set_params(bool fit_intercept = true, bool normalize = false, bool copy_X = true, int n_jobs = 1, bool ols = false, int epochs = 1000);
};

Matrix LinearRegression::fit(Matrix X, Matrix Y){ //estimate coefficients
/* 

	// add bias factor
	
	//ols
	if (ols){
		Matrix C = ops.inverse((ops.matrix_mul(X.transpose(), X)));
		Matrix D = ops.matrix_mul(X.transpose(), Y)
		B = ops.matrix_mul(C,D);
		return B;
	}

	// gradient descent
		else{
		double learning_rate = 0.001;
		for ( int i = 1 ; i <= epochs ; i++ ){
			Matrix x_transpose_x = ops.matrix_mul(X.transpose(), X);
			Matrix x_transpose_y = ops.matrix_mul(X.transpose(), Y);
			B = B - (2/epochs)*(learning_rate)*(ops.matrix_mul(x_transpose_x, B) - x_transpose_y);
			B_dash = B_dash - (learning_rate)*()
		}
		return std::make_pair(B, B_dash); 
*/

}

#endif /* _linear_regression_hpp_ */