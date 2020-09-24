#ifndef _model_selection_hpp_
#define _model_selection_hpp_

#include <all.hpp>

class ModelSelection {
  private:
  public:
    std::tuple<Matrix, Matrix, Matrix, Matrix> train_test_split(Matrix, Matrix, int, float, float);
} model_selection;

std::tuple<Matrix, Matrix, Matrix, Matrix>
ModelSelection::train_test_split(Matrix X, Matrix y, int random_state = -1, float train_size = 0.75,
                                 float test_size = 0.25) {
    if (train_size + test_size > 1) {
        assert(("Size arguments are wrong.", false));
    }

    if (random_state >= 0) {
        srand(random_state);
    } else {
        srand(time(NULL));
    }

    std::vector<std::vector<double>> X_train_vec, X_test_vec, y_train_vec, y_test_vec;

    // Assumption: Examples are rows in the Matrix
    int n = X.row_length();
    for (int i = 0; i < n; i++) {
        float num = ((float)(rand() % n)) / n;
        if (num < train_size) {
            X_train_vec.push_back(X.get_row(i));
            y_train_vec.push_back(y.get_row(i));
        } else {
            X_test_vec.push_back(X.get_row(i));
            y_test_vec.push_back(y.get_row(i));
        }
    }

    Matrix X_train = matrix.init(X_train_vec);
    Matrix y_train = matrix.init(y_train_vec);
    Matrix X_test = matrix.init(X_test_vec);
    Matrix y_test = matrix.init(y_test_vec);

    return {X_train, X_test, y_train, y_test};
}

#endif /* _model_selection_h_ */