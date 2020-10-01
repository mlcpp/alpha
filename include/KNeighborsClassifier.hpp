#ifndef _k_neighbors_classifier_hpp_
#define _k_neighbors_classifier_hpp_

#include <all.hpp>

class KNeighborsClassifier {
private:
	Matrix distance(Matrix, Matrix);
	bool is_fit;
	int n_clusters, epochs;
	long double score;

public:
	KNeighborsClassifier (int, int);
	void fit(Matrix);
	void get_params();
	long double score_();
	Matrix KNeighbors(double x, double y);
	Matrix predict(Matrix);
	Matrix predict_proba(Matrix);
	void set_params(int, int);
};

// Constructor
KNeighborsClassifier::KNeighborsClassifier(int n_clusters = 3, int epochs = 100) {
	this->n_clusters = n_clusters;
    this->epochs = epochs;
    is_fit = false;
}

// Method to fit
void KNeighborsClassifier::fit(Matrix X) {

	is_fit = true;
}

// Method to calculate the score
long double KNeighborsClassifier::score_() {

	return score;
}

// Method to
Matrix KNeighborsClassifier::predict(Matrix X) {
	assert(("Fit the model before predicting.", is_fit));
	return X;
}

// Method to print the KNeighborsClassifier object parameters in json format
void KNeighborsClassifier::get_params() {
	std::cout << std::boolalpha;
    std::cout << "[" << std::endl;
    std::cout << "\t \"n_clusters\": \"" << n_clusters << "\"," << std::endl;
    std::cout << "\t \"epochs\": \"" << epochs << "\"" << std::endl;
    std::cout << "]" << std::endl;
}

// Method to set the KNeighborsClassifier object parameters
void KNeighborsClassifier::set_params(int n_clusters = 3, int epochs = 100) {
	this->n_clusters = n_clusters;
    this->epochs = epochs;
}

// Helper methods

// Method to calculate square of Euclidean Distance
Matrix KNeighborsClassifier::distance(Matrix X, Matrix C) {
	std::vector<std::vector<double>> res;
	for (int i = 0; i < X.row_length(); i++) {
		std::vector<double> row;
		for (int j = 0; j < C.row_length(); j++) {
			Matrix sum = matrix.sum(matrix.power(X.slice(i, i + 1, 0, X.col_length()) -
							     C.slice(j, j + 1, 0, C.col_length()),
							     2),
						"row");
			row.push_back(sum(0, 0));
		}
		res.push_back(row);
		row.clear();
	}
	return matrix.init(res);
}

#endif /* _k_neighbors_classifier_hpp_ */
