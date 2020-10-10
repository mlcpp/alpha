#ifndef _preprocessing_hpp_
#define _preprocessing_hpp_

#include <all.hpp>

class Preprocessing {
  private:
    bool if_mean = false, if_std = false;
    Matrix mean, std;

  public:
    Matrix normalize(Matrix, std::string);
    ~Preprocessing() {}
};

// Method to normalize the dataset
Matrix Preprocessing::normalize(Matrix mat, std::string dim) {
    if (!if_mean) {
        mean = matrix.mean(mat, dim);
        if_mean = true;
    }
    if (!if_std) {
        std = matrix.std(mat, dim);
        if_std = true;
    }
    Matrix result = (mat - mean) / std;
    return result;
}

#endif /* _preprocessing_h_ */
