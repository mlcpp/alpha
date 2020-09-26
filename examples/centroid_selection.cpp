#include <all.hpp>
#include <Matrix.hpp>
using namespace std;

Matrix centroid_selection(Matrix X, int k) {
    std::vector<std::vector<double>> temp_vec;
    srand(time(NULL));

    for (int i = 0; i < k; i++) {
        temp_vec.push_back(X.get_row(rand() % X.row_length()));
    }
    return matrix.init(temp_vec);
}

int main(){

    Matrix mat = read_csv("./datasets/iris/iris.csv");
    Matrix X = mat.slice(1, 150, 0, 2);
    X.to_double();
    Matrix C = centroid_selection(X, 3);
    C.print();
}
