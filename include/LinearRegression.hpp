#ifndef _linear_regression_hpp_
#define _linear_regression_hpp_

template <typename T>
class LinearRegression{
  private:
  public:
  // void fit(Matrix X);
  void fit(Matrix X, Matrix Y);
  void get_params();
  void predict(Matrix X);
  void score(Matrix X, Matrix Y);
  void set_params();
};

#endif /* _linear_regression_hpp_ */