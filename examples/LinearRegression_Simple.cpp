// Header files
#include <LinearRegression.hpp>
#include <matplotlibcpp.hpp>
namespace plt = matplotlibcpp;

int main(){
 
    // Load the datasets
    Matrix mat = read_csv("./datasets/boston/diabetes.csv");
/*
    // Use only one feature
    mat = mat.slice();
    mat.to_double();

    // Split the data into training/testing sets
    pair <Matrix, Matrix> X = split_test_train(mat);
    Matrix X_train = X.first;
    Matrix X_test = X.second;

    // Split the targets into training/testing sets
    pair <Matrix, Matrix> Y = split_test_train(mat[1]);
    Matrix Y_train = Y.first;
    Matrix Y_test = Y.second;
 */
    // Create linear regression object
    LinearRegression regr;
    regr.get_params();
    regr.set_params(false, false, false,10, false, 1);
    regr.get_params();

/*     // Train the model using the training set
    regr.fit(X_train, Y_train);

    // Make prediction using the testing set
    Matrix Y_pred = regr.predict(X_test);

    // Print coefficients
    Matrix coef = regr.coef_;
    coef.print();

    // The mean squared error
    std :: cout << mean_squared_error(diabetes_y_test, diabetes_y_pred);

    // The coefficient of determination: 1 is perfect prediction
    std :: cout << r2_score(diabetes_y_test, diabetes_y_pred);

    // Plot outputs
    plt::figure_size(1200, 780);
    plt::scatter(X_test, Y_test);
    plt::plot(X_test, Y_pred);
    plt::named_plot("Linear regression", x, y);
    // Set x-axis to interval [0,1000000]
    plt::xlim(0, 1000*1000);
    // Add graph title
    plt::title("Sample figure");
    // Enable legend.
    plt::legend();
    // Save the image (file format is determined by the extension)
    plt::save("./build/LinearRegression.png");

    plt::show();
 */
    return 0;
}