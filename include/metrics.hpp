#ifndef _metrics_hpp_
#define _metrics_hpp_

#include <all.hpp>

class metrics {
  public:
    // Classification metrics
    long double accuracy_score(Matrix, Matrix, bool);
    Matrix confusion_matrix(Matrix, Matrix, int);
    long double f1_score(Matrix, Matrix, int);
    long double precision_score(Matrix, Matrix, int);
    long double recall_score(Matrix, Matrix, int);

    // Regresssion metrics
    long double mean_absolute_error(Matrix, Matrix);
    long double mean_squared_error(Matrix, Matrix);
    double r2_score(Matrix, Matrix);
    ~metrics() {}
} metrics;

// Classification metrics

// Return accuracy score
long double metrics::accuracy_score(Matrix y_true, Matrix y_pred, bool normalize = true) {
    long double count = 0;
    for (int i = 0; i < y_true.row_length(); i++) {
        if (y_true(i, 0) == y_pred(i, 0)) {
            count++;
        }
    }

    if (normalize)
        return count / y_true.row_length();
    else
        return count;
}

// Return a confusion matrix where rows are true labels and columns are predictions
Matrix metrics::confusion_matrix(Matrix y_true, Matrix y_pred, int num_labels = 2) {
    std::vector<std::vector<double>> res;
    for (int i = 0; i < num_labels; i++) {
        std::vector<double> row(num_labels, 0);
        Matrix y_pred_i = matrix.slice_select(y_pred, y_true, i, 0);
        for (int j = 0; j < y_pred_i.row_length(); j++)
            row[y_pred_i(j, 0)] += 1;
        res.push_back(row);
    }

    return matrix.init(res);
}

// Return F1 score
long double metrics::f1_score(Matrix y_true, Matrix y_pred, int num_labels = 2) {
    long double precision = precision_score(y_true, y_pred, num_labels);
    long double recall = recall_score(y_true, y_pred, num_labels);
    long double f1 = 2 * precision * recall / (precision + recall);

    return f1;
}

// Return precision score
// For multiclass it returns average of precision score for each class
long double metrics::precision_score(Matrix y_true, Matrix y_pred, int num_labels = 2) {
    if (num_labels == 2) {
        Matrix con_mat = confusion_matrix(y_true, y_pred);
        return con_mat(1, 1) / (con_mat(1, 1) + con_mat(0, 1));
    }
    long double precision = 0;
    for (int i = 0; i < num_labels; i++) {
        Matrix true_p = matrix.slice_select(y_true, y_true, i, 0);
        Matrix pred_p = matrix.slice_select(y_pred, y_true, i, 0);
        long long tp = matrix.slice_select(true_p, pred_p, i, 0).row_length();
        precision += (long double)tp / pred_p.row_length();
    }

    return precision / num_labels;
}

// Return recall score
// For multiclass it returns average of recall score for each class
long double metrics::recall_score(Matrix y_true, Matrix y_pred, int num_labels = 2) {
    if (num_labels == 2) {
        Matrix con_mat = confusion_matrix(y_true, y_pred);
        return con_mat(1, 1) / (con_mat(1, 1) + con_mat(1, 0));
    }
    long double recall = 0;
    for (int i = 0; i < num_labels; i++) {
        Matrix true_p = matrix.slice_select(y_true, y_true, i, 0);
        Matrix pred_p = matrix.slice_select(y_pred, y_true, i, 0);
        long long tp = matrix.slice_select(true_p, pred_p, i, 0).row_length();
        recall += (long double)tp / true_p.row_length();
    }

    return recall / num_labels;
}

// Regression metrics

// Return mean absolute error regression loss
long double metrics::mean_absolute_error(Matrix y_true, Matrix y_pred) {
    Matrix diff = y_pred - y_true;
    long double score = matrix.sum(matrix.abs(diff), "column")(0, 0) / y_true.row_length();

    return score;
}

// Return mean squared error regression loss
long double metrics::mean_squared_error(Matrix y_true, Matrix y_pred) {
    Matrix diff = y_pred - y_true;
    long double score = matrix.matmul(diff.T(), diff)(0, 0) / y_true.row_length();

    return score;
}

// Return R^2 (coefficient of determination) score
double metrics::r2_score(Matrix y_true, Matrix y_pred) {
    double y_mean = ((matrix.mean(y_true, "column"))(0, 0));
    double residual_sum_of_squares =
        (matrix.matmul((y_pred - y_true).T(), (y_pred - y_true)))(0, 0);
    double total_sum_of_squares = (matrix.matmul((y_true - y_mean).T(), (y_true - y_mean)))(0, 0);

    double score = (1 - (residual_sum_of_squares / total_sum_of_squares));
    return score;
}

#endif /* _metrics_h_ */
