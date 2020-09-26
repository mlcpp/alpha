#ifndef _k_means_hpp_
#define _k_means_hpp_

#include <all.hpp>

class KMeans {
  private:
    int n_clusters, epochs;
    Matrix centroid_selection(Matrix, int);
    Matrix distance(Matrix, Matrix);

  public:
    KMeans(int n_clusters, int epochs) {
        this->n_clusters = n_clusters;
        this->epochs = epochs;
    }
    // computes k optimal centroids and classifies given X points
    void fit(Matrix);

    // compute k optimal centroids and classifies each new points into k clusters
    Matrix fit_predict(Matrix);

    Matrix fit_transform(Matrix);
    void get_params();

    // classifies new given points into one of ready k clusters
    Matrix predict(Matrix);
    double score(Matrix);
    void set_params();
    Matrix predict(Matrix);
    Matrix transform(Matrix);
};

void KMeans::centroid_selection(Matrix X, int k) {
    std::vector<std::vector<double>> temp_vec;
    srand(time(NULL));

    for (int i = 0; i < k; i++) {
        temp_vec.push_back(X.get_row(rand() % X.row_length()));
    }
    return matrix.init(temp_vec);
}

void KMeans::distance(Matrix X, Matrix Y){
  // check their dimensions are same
  return matrix.sqrt(matrix.power(X-Y, 2));
}

void KMeans::fit(Matrix X) {
    /*
     * input mxn matrix
     * return matrix Z where Z[i][j] = 1 if j belongs to cluster i
     * either run for epochs, or stop at convergence, ie z - z' = 0
     */

    Matrix C = centroid_selection(X, k);
    Matrix Z = matrix.zeros(X.row_length(), k);

    Matrix Y = Y - matrix.matmul(( X - matrix.matmul(Z.T(), C), ( X - matrix.matmul(Z.T(), C).T());
  	Matrix C = matrix.matmul(matrix.matmul(X, Z.T()), matrix.inverse(matrix.matmul(Z, Z.T())))
	
  	Matrix Z = matrix.zeros(X.row_length(), 1);
  	for ( int i = 0 ; i < epochs ; i++ ){
        Matrix temp = distance(X, C);         // (m,k)
        Matrix z = matrix.argmin(temp, "row") // (m,1)
                   C = update_mean(X, C, Z);
	}
}

Matrix<>::update_mean() {
    for (int j = 0; j < k; j++) {
        C = update_mean(all x where x belongs to ci) // (k, n)
    }

#endif /* _k_means_hpp_ */