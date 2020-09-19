#ifndef _linear_regression_hpp_
#define _linear_regression_hpp_

#include <all.h>

template <typename T>
class LinearRegression {
  private:
  public:
    void fit();
    void get_params();
    void predict();
    void score();
    void set_params();
};

#endif /* _linear_regression_hpp_ */