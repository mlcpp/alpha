#ifndef _pca_hpp_
#define _pca_hpp_

#include <all.hpp>

class PCA {
  private:
    bool is_fit = false;
    int n_components;
    Matrix Ureduce, Sigma;

    Eigen::MatrixXd convert2eigen(Matrix);
    Matrix convert2matrix(Eigen::MatrixXd);

  public:
    PCA(int);
    void fit(Matrix);
    Matrix fit_transform(Matrix);
    Matrix get_covariance();
    void get_params();
    Matrix get_precision();
    Matrix inverse_transform(Matrix);
    double score(Matrix);
    Matrix score_samples(Matrix);
    void set_params(int);
    Matrix transform(Matrix);
};

// Constructor
PCA::PCA(int n_components = -1) { this->n_components = n_components; }

// Method to fit the PCA model
void PCA::fit(Matrix X) {
    if (n_components == -1)
        n_components = X.col_length();

    Sigma = matrix.matmul(X.T(), X) / (X.row_length());
    Eigen::MatrixXd Sigma_m = convert2eigen(Sigma);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Sigma_m, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Matrix U = convert2matrix(svd.matrixU());
    Ureduce = U.slice(0, U.row_length(), 0, n_components);

    is_fit = true;
}

// Method to fit the PCA model and then tranform given Matrix object
Matrix PCA::fit_transform(Matrix X) {
    fit(X);
    Matrix Z = transform(X);
    return Z;
}

// Method to get the Covariance Matrix
Matrix PCA::get_covariance() { return Sigma; }

// Method to get the Precision Matrix
Matrix PCA::get_precision() { return matrix.inverse(Sigma); }

// Method to print the PCA object parameters in json format
void PCA::get_params() {
    std::cout << std::boolalpha;
    std::cout << "[" << std::endl;
    std::cout << "\t \"n_components\": \"" << n_components << "\"" << std::endl;
    std::cout << "]" << std::endl;
}

// Method to tranform the data back to original dimensions
Matrix PCA::inverse_transform(Matrix Z) {
    Matrix X_recovered = (matrix.matmul(Ureduce, Z.T())).T();
    return X_recovered;
}

double PCA::score(Matrix X) { return matrix.mean(score_samples(X), "row")(0, 0); }

Matrix PCA::score_samples(Matrix X) {
    Matrix Xr = X - matrix.mean(X, "column");
    int n_features = X.col_length();
    Matrix precision = get_precision();
    Matrix log_like = matrix.sum((Xr * (matrix.matmul(Xr, precision))), "column") * -0.5;
    log_like = log_like - (((log(2 * 3.14) * n_features) -
                            log(matrix.determinant(precision, precision.row_length()))) *
                           0.5);

    return log_like;
}

// Method to set the PCA object parameters
void PCA::set_params(int n_components = -1) { this->n_components = n_components; }

Matrix PCA::transform(Matrix X) {
    assert(("Fit the model before predicting.", is_fit));

    Matrix Z = matrix.matmul(X, Ureduce);

    return Z;
}

// Helper methods

// Helper method to convert a Matrix object to Eigne::MatrixXd object
Eigen::MatrixXd PCA::convert2eigen(Matrix mat) {
    Eigen::MatrixXd res(mat.row_length(), mat.col_length());
    for (int i = 0; i < mat.row_length(); i++) {
        for (int j = 0; j < mat.col_length(); j++)
            res(i, j) = mat(i, j);
    }

    return res;
}

// Helper method to convert a MatrixXd object to Eigne::Matrix object
Matrix PCA::convert2matrix(Eigen::MatrixXd m) {
    // Eigen::MatrixXd res(mat.row_length(), mat.col_length());
    std::vector<std::vector<double>> res;
    std::vector<double> row;
    for (int i = 0; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++)
            row.push_back(m(i, j));
        res.push_back(row);
        row.clear();
    }

    return matrix.init(res);
}

#endif /* _pca_hpp_ */
