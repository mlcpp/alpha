#ifndef _k_means_hpp_
#define _k_means_hpp_

#include <all.hpp>

class KMeans {
  private:
    int n_clusters, epochs;
    Matrix C;
    Matrix centroid_selection(Matrix, int);
    Matrix distance(Matrix, Matrix);
    Matrix update_centroid(Matrix, Matrix, Matrix);
    bool is_fit;
    long double J;

  public:
    KMeans(int, int);
    void fit(Matrix);
    Matrix fit_predict(Matrix);
    void get_params();
    long double score();
    Matrix predict(Matrix);
    double score(Matrix);
    void set_params(int, int);
    Matrix get_centroid();
};

KMeans::KMeans(int n_clusters = 3, int epochs = 100) {
    this->n_clusters = n_clusters;
    this->epochs = epochs;
    J = 0;
    is_fit = 0;
}

Matrix KMeans::centroid_selection(Matrix X, int k) {
    std::vector<std::vector<double>> temp_vec;
    srand(time(NULL));

    for (int i = 0; i < k; i++) {
        temp_vec.push_back(X.get_row(rand() % X.row_length()));
    }
    return matrix.init(temp_vec);
}

Matrix KMeans::distance(Matrix X, Matrix C) {
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

// computes k optimal centroids and classifies given X points
void KMeans::fit(Matrix X) {
    C = centroid_selection(X, n_clusters);
    for (int i = 0; i < epochs; i++) {
        Matrix temp = distance(X, C);
        Matrix Z = matrix.argmin(temp, "row");
        C = update_centroid(X, C, Z);
    }
    is_fit = true;
}

Matrix KMeans::update_centroid(Matrix X, Matrix C, Matrix Z) {
    std::vector<std::vector<std::vector<double>>> cluster_members;
    for (int i = 0; i < C.row_length(); i++) {
        std::vector<std::vector<double>> rows;
        cluster_members.push_back(rows);
    }

    for (int i = 0; i < Z.row_length(); i++) {
        cluster_members[Z(i, 0)].push_back(X.get_row(i));
    }
    assert(("K is more than the current number of clusters.", cluster_members[0].size() != 0));
    J += matrix.sum(
        matrix.sqrt(distance(matrix.init(cluster_members[0]), C.slice(0, 1, 0, C.col_length()))),
        "column")(0, 0);
    Matrix X_mean = matrix.mean(matrix.init(cluster_members[0]), "column");
    for (int i = 1; i < C.row_length(); i++) {
        assert(("K is more than the current number of clusters.", cluster_members[i].size() != 0));
        X_mean =
            matrix.concat(X_mean, matrix.mean(matrix.init(cluster_members[i]), "column"), "row");
        J += matrix.sum(matrix.sqrt(distance(matrix.init(cluster_members[1]),
                                             C.slice(i, i + 1, 0, C.col_length()))),
                        "column")(0, 0);
    }
    J = J / X.row_length();
    return X_mean;
}

long double KMeans::score() {
    long double score = -1 * J;
    return score;
}

// classifies new given points into one of ready k clusters
Matrix KMeans::predict(Matrix X) {
    Matrix Z = matrix.zeros(X.row_length(), 1);
    Matrix temp = distance(X, C);
    Z = matrix.argmin(temp, "row");
    return Z;
}

// compute k optimal centroids and classifies each new points into k clusters
Matrix KMeans::fit_predict(Matrix X) {
    fit(X);
    Matrix Y_pred = predict(X_test);
    return Y_pred;
}

Matrix KMeans::get_centroid() { return C; }

// Method to print the KMeans object parameters in json format
void KMeans::get_params() {
    std::cout << std::boolalpha;
    std::cout << "[" << std::endl;
    std::cout << "\t \"n_clusters\": \"" << n_clusters << "\"," << std::endl;
    std::cout << "\t \"epochs\": \"" << epochs << "\"" << std::endl;
    std::cout << "]" << std::endl;
}

// Method to set the KMeans object parameters
void KMeans::set_params(int n_clusters = 3, int epochs = 100) {
    this->n_clusters = n_clusters;
    this->epochs = epochs;
}

#endif /* _k_means_hpp_ */
