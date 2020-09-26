#ifndef _k_means_hpp_
#define _k_means_hpp_

#include <all.hpp>

class KMeans {
  private:
    int n_clusters, epochs;
    Matrix C;
    Matrix centroid_selection(Matrix, int);
    Matrix distance(Matrix, Matrix);
    Matrix update_mean(Matrix, Matrix, Matrix);

  public:
    KMeans(int, int);
    void fit(Matrix);
    Matrix fit_predict(Matrix);
    void get_params();
    Matrix predict(Matrix);
    double score(Matrix);
    void set_params();
    Matrix get_centroid();
};

KMeans::KMeans(int n_clusters = 3, int epochs = 1000) {
    this->n_clusters = n_clusters;
    this->epochs = epochs;
}

Matrix KMeans::centroid_selection(Matrix X, int k) {
    std::vector<std::vector<double>> temp_vec;
    srand(time(NULL));

    for (int i = 0; i < k; i++) {
        temp_vec.push_back(X.get_row(rand() % X.row_length()));
    }
    return matrix.init(temp_vec);
}

Matrix KMeans::distance(Matrix X, Matrix Y) {
    // check their dimensions are same
    if ((X.col_length() != Y.col_length()) || (X.row_length() != Y.row_length())) {
        assert(("The Matrix objects should be of same dimensions", false));
    }

    Matrix sqr = matrix.power(X - Y, 2);
    std::vector<std::vector<double>> vec = sqr.get();

    // reduce sqr matrix to 1D
    for (int i = 0; i < sqr.row_length(); i++) {
        for (int j = 1; j < sqr.col_length(); j++) {
            vec[i][0] += vec[i][j];
        }
    }
    Matrix dist = matrix.init(vec);
    dist = matrix.sqrt(dist.slice(0, dist.row_length(), 0, 1));

    return dist;
}
// computes k optimal centroids and classifies given X points
void KMeans::fit(Matrix X) {
    C = centroid_selection(X, n_clusters);
    Matrix Z = matrix.zeros(X.row_length(), 1);
    for (int i = 0; i < epochs; i++) {
        Matrix temp = distance(X, C);          // (m,k)
        Matrix Z = matrix.argmin(temp, "row"); // (m,1)
        C = update_mean(X, C, Z);
    }
}

Matrix KMeans::update_mean(Matrix X, Matrix C, Matrix Z) {
    std::vector<std::vector<std::vector<double>>> cluster_members;
    for (int i = 0; i < C.row_length(); i++) {
        std::vector<std::vector<double>> rows;
        cluster_members.push_back(rows);
    }
    for (int i = 0; i < Z.row_length(); i++) {
        cluster_members[Z(i, 0)].push_back(X.get_row(i));
    }
    Matrix X_mean = matrix.mean(matrix.init(cluster_members[0]), "column");
    for (int i = 1; i < C.row_length(); i++) {
        X_mean =
            matrix.concat(X_mean, matrix.mean(matrix.init(cluster_members[i]), "column"), "row");
    }
    return X_mean;
}

// classifies new given points into one of ready k clusters
Matrix KMeans::predict(Matrix X) {
    Matrix Z = matrix.zeros(X.row_length(), 1);
    Matrix temp = distance(X, C); // (m,k)
    Z = matrix.argmin(temp, "row");
    return Z;
}

// compute k optimal centroids and classifies each new points into k clusters
Matrix KMeans::fit_predict(Matrix X_test) {
    fit(X_test);
    Matrix X_pred = predict(X_test);
    return X_pred;
}

Matrix get_centroid(){
    return C;
}

#endif /* _k_means_hpp_ */