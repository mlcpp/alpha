#ifndef _k_means_hpp_
#define _k_means_hpp_

#include <all.hpp>

class KMeans {
  private:
    int n_clusters, epochs;

  public:
    KMeans(int n_clusters, int epochs) {
        this->n_clusters = n_clusters;
        this->epochs = epochs;
    }
    // computes k optimal centroids and classifies given X points
    Matrix fit(Matrix X);

    // compute k optimal centroids and classifies each new points into k clusters
    Matrix fit_predict(Matrix X);

    Matrix fit_transform(Matrix X);
    void get_params();

    // classifies new given points into one of ready k clusters
    Matrix predict(Matrix X);
    double score(Matrix X);
    void set_params();
    Matrix predict(Matrix X);
    Matrix transform(Matrix X);
};

Matrix KMeans::fit(Matrix X) {
    /*
     * input mxn matrix
     * return matrix Y where Y[i][j] = 1 if j belongs to cluster i
     */
}

#endif /* _k_means_hpp_ */