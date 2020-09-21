#ifndef _linear_regression_hpp_
#define _linear_regression_hpp_

#include <all.hpp>

class LinearRegression{
	private:
		bool fit_intercept = true, normalize=false,  copy_X = true, ols = false;
		int n_jobs = 1, epochs = 1000;
		Matrix B;
	public:
/* 	LinearRegression(){
		fit_intercept = true;
		normalize = false; 
		copy_X = true; 
		ols = false;
		n_jobs = 1;
		epochs = 1000;
	} */
	LinearRegression(bool fit_intercept = true, bool normalize = false, bool copy_X = true, int n_jobs = 1, bool ols = false, int epochs = 1000){
		this->fit_intercept = fit_intercept;
		this->normalize = normalize;
		this->copy_X = copy_X;
		this->ols = ols;
		this->n_jobs = n_jobs;
		this->epochs = epochs;
	}
	Matrix fit(Matrix X, Matrix Y);
	void get_params(); // returns a map //prints json format
	void predict(Matrix X);
	void score(Matrix X, Matrix Y);
	void set_params(bool fit_intercept = true, bool normalize = false, bool copy_X = true, int n_jobs = 1, bool ols = false, int epochs = 1000);
};


Matrix LinearRegression::fit(Matrix X, Matrix Y){ //estimate coefficients
	// add bias factor
	
	//ols
	if (ols){
		Matrix C = (ops.matrix_mult(X.T(), X)).inverse();
		Matrix D = ops.matrix_mult(X.T(), Y);
		B = ops.matrix_mult(C,D);
		return B;
	}

	// gradient descent
	else{
		double learning_rate = 0.001;
		for ( int i = 1 ; i <= epochs ; i++ ){
			Matrix x_transpose_x = ops.matrix_mult(X.T(), X);
			Matrix x_transpose_y = ops.matrix_mult(X.T(), Y);
			B = B - (ops.matrix_mult(Y.T()-Y, X))*(learning_rate);
		}
		return B;
	}

}

void LinearRegression::get_params(){
	std::cout << std::boolalpha;   
	std::vector<std::string> params = { "fit_intercept", "normalize", "copy_X", "ols", "n_jobs", "epochs" };
	std::cout << "[" <<std::endl;
	std::cout << "\t \"fit_intercept\": \""<<this->fit_intercept<<"\","<<std::endl;
	std::cout << "\t \"normalize\": \""<<this->normalize<<"\","<<std::endl;
	std::cout << "\t \"copy_X\": \""<<this->copy_X<<"\","<<std::endl;
	std::cout << "\t \"ols\": \""<<this->ols<<"\","<<std::endl;
	std::cout << "\t \"n_jobs\": \""<<this->n_jobs<<"\","<<std::endl;
	std::cout << "\t \"epochs\": \""<<this->epochs<<"\""<<std::endl;
	std::cout << "]" <<std::endl;
}

void LinearRegression::set_params(bool fit_intercept, bool normalize, bool copy_X, int n_jobs, bool ols, int epochs){
		this->fit_intercept = fit_intercept;
		this->normalize = normalize;
		this->copy_X = copy_X;
		this->ols = ols;
		this->n_jobs = n_jobs;
		this->epochs = epochs;
}

#endif /* _linear_regression_hpp_ */