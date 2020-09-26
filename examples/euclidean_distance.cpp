#include <KMeans.hpp>
#include <Matrix.hpp>
#include <all.hpp>
using namespace std;

int main() {

    Matrix mat = read_csv("./datasets/iris/iris.csv");
    Matrix X = mat.slice(1, 2, 0, 2);
    Matrix Y = mat.slice(1, 2, 2, 4);
    X.to_double();
    Y.to_double();
    X.print();
    Y.print();
    KMeans k;
}
