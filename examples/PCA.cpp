// Include necessary header files
#include <Matrix.hpp>
#include <PCA.hpp>
#include <preprocessing.hpp>

// Example program

// Read csv files to get a Matrix object.
// Slice the Matrix object to remove the header.
// Fitting and transforming the dataset using the PCA model.
int main() {
    // Load the dataset
    Matrix mat = read_csv("./datasets/blobs_pca/blobs_pca.csv");

    // Remove the header description from the dataset
    Matrix X = mat.slice(1, mat.row_length(), 0, mat.col_length());

    // Convert the data from string to double
    X.to_double();

    // Normalizing the dataset
    Preprocessing preprocessing;
    X = preprocessing.normalize(X, "column");

    // PCA to project 2 features onto 1

    // Create PCA object
    std::cout << "PCA to project 2 features onto 1: " << std::endl << std::endl;

    PCA pca(1);

    // Fitting and transforming the dataset
    Matrix X_transform = pca.fit_transform(X);

    std ::cout << "Score: " << pca.score(X) << std::endl;

    std::cout << std::endl;

    // Printing the transformed X
    X_transform.print();

    return 0;
}
