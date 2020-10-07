#ifndef _k_neighbors_classifier_hpp_
#define _k_neighbors_classifier_hpp_

#include <all.hpp>

class KNeighborsClassifier {
  private:
    Matrix distance(Matrix, Matrix);
    bool is_fit;
    int n_neighbors;
    Matrix X, Y, labels;
    std::vector<double> select_vec(Matrix, int);

  public:
    KNeighborsClassifier(int);
    void fit(Matrix, Matrix);
    void get_params();
    Matrix kneighbors(std::vector<double>);
    Matrix predict(Matrix);
    double score(Matrix, Matrix);
    void set_params(int);
    ~KNeighborsClassifier() {}
};

// Constructor
KNeighborsClassifier::KNeighborsClassifier(int n_neighbors = 3) {
    this->n_neighbors = n_neighbors;
    is_fit = false;
}

// Method to fit the KNeighboursClassifier model
void KNeighborsClassifier::fit(Matrix X, Matrix Y) {
    this->X = X;
    this->Y = Y;
    is_fit = true;
}

// Method to print the KNeighborsClassifier object parameters in json format
void KNeighborsClassifier::get_params() {
    std::cout << std::boolalpha;
    std::cout << "[" << std::endl;
    std::cout << "\t \"n_neighbors\": \"" << n_neighbors << " \" " << std::endl;
    std::cout << "]" << std::endl;
}

// Method to predict the KNeighbours of a single point
Matrix KNeighborsClassifier::kneighbors(std::vector<double> P) {
    assert(("Fit the model before predicting.", is_fit));
    std::vector<std::vector<double>> vec;
    vec.push_back(P);
    Matrix X = matrix.init(vec);
    Matrix dist = distance(this->X, X.slice(0, 1, 0, X.col_length()));
    std::vector<std::vector<double>> res;
    res.push_back(select_vec(dist, n_neighbors));
    labels = matrix.init(res);
    return labels;
}

// Method to predict the KNeighbors of a matrix
Matrix KNeighborsClassifier::predict(Matrix X) {
    std::vector<std::vector<double>> res;
    assert(("Fit the model before predicting.", is_fit));
    for (int i = 0; i < X.row_length(); i++) {
        Matrix dist = distance(this->X, X.slice(i, i + 1, 0, X.col_length()));
        res.push_back(select_vec(dist, n_neighbors));
    }
    labels = matrix.init(res);
    return labels;
}

// Method to calculate the score of the model
double KNeighborsClassifier::score(Matrix Y_pred, Matrix Y_true) {
    double count = 0;
    for (int i = 0; i < Y_true.row_length(); i++) {
        if (Y_true(i, 0) == Y_pred(i, 0)) {
            count++;
        }
    }
    return count / Y_true.row_length();
}

// Method to set the KNeighborsClassifier object parameters
void KNeighborsClassifier::set_params(int n_neighbors = 3) { this->n_neighbors = n_neighbors; }

// Helper methods

// Method to calculate square of Euclidean Distance
Matrix KNeighborsClassifier::distance(Matrix X, Matrix C) {
    std::vector<std::vector<double>> res;
    for (int i = 0; i < X.row_length(); i++) {
        std::vector<double> row;
        Matrix sum = matrix.sum(
            matrix.power(X.slice(i, i + 1, 0, X.col_length()) - C.slice(0, 1, 0, C.col_length()),
                         2),
            "row");
        row.push_back(sum(0, 0));
        res.push_back(row);
        row.clear();
    }
    return matrix.init(res);
}

std::vector<double> KNeighborsClassifier::select_vec(Matrix dist, int k) {
    // tuple of Y and dist
    std::vector<std::tuple<double, double>> v_t;
    for (int i = 0; i < dist.row_length(); i++) {
        v_t.push_back(std::make_tuple(dist(i, 0), Y(i, 0)));
    }
    // sort tuple<dist, Y> keeping Y intact
    sort(v_t.begin(), v_t.end());
    std::vector<double> row;
    std::vector<bool> visited(k, false);
    int max_count = 0;
    double val = 0.0;
    // select first k rows
    for (int i = 0; i < k; i++) {
        // find the majority of rows using Y labels
        if (visited[i] == true) {
            continue;
        }
        int count = 0;
        for (int j = 0; j < k; j++) {
            if (Y(i, 0) == Y(j, 0)) {
                visited[j] = true;
                count++;
            }
        }
        if (count > max_count) {
            max_count = count;
            val = std::get<1>(v_t[i]);
        }
    }
    row.push_back(val);
    return row;
}

#endif /* _k_neighbors_classifier_hpp_ */
