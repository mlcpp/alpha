#ifndef _preprocessing_hpp_
#define _preprocessing_hpp_

#include <all.hpp>

class Preprocessing {
  private:
  public:
    Matrix normalize(Matrix, std::string);
} preprocessing;

Matrix Preprocessing::normalize(Matrix mat, std::string dim) {
    Matrix result = (mat - matrix.mean(mat, dim)) / matrix.std(mat, dim);
    return result;
}

#endif /* _preprocessing_h_ */