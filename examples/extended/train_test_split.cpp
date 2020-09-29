#include <Matrix.hpp>
#include <model_selection.hpp>

/* Example program

Read csv files to get a Matrix object.
Slice the Matrix object according to our needs.
Then we can split our dataset using train_test_split.
*/
int main() {
    Matrix mat = read_csv("./datasets/diabetes/diabetes.csv");

    Matrix X = mat.slice(1, 30, 0, mat.col_length() - 1);
    X.to_double();
    Matrix y = mat.slice(1, 30, mat.col_length() - 1, mat.col_length());
    y.to_double();

    // Normalizing the Matrix object
    auto [X_train, X_test, y_train, y_test] = model_selection.train_test_split(X, y, 0);

    std::cout << "X_train: " << std::endl;
    X_train.print();
    std::cout << std::endl;

    std::cout << "X_test: " << std::endl;
    X_test.print();
    std::cout << std::endl;

    std::cout << "y_train: " << std::endl;
    y_train.print();
    std::cout << std::endl;

    std::cout << "y_test: " << std::endl;
    y_test.print();
    std::cout << std::endl;

    return 0;
}