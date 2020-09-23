// import header files
#include <KMeans.hpp>
#include <matplotlibcpp.hpp>
namespace plt = matplotlibcpp;

// create dataset with two feature
Matrix X = read_csv("./datasets/make_blob/make_blob.csv");

// plot uncategorized data
plt.scatter(

   X.slice(: , 0), X.slice(: , 1),
   c='white', marker='o',
   edgecolor='black', s=50
)
plt.show()

// create KMeans object
Matrix km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)

Matrix Y_pred = km.fit_predict(X) // fit() and predict()

// plot the 3 clusters (categorized data)
plt.scatter(
    // (first feature where y = 0, second feature)
    X.slice(Y_pred == 0, 0), X.slice(Y_pred == 0, 1), // X[y==0, 0]
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X.slice(Y_pred == 1, 0), X.slice(Y_pred == 1, 1),
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X.slice(Y_pred == 2, 0), X.slice(Y_pred == 2, 1),
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

// also plot the centroids of the categorized data
plt.scatter(
    km.cluster_centers().slice(:, 0), km.cluster_centers().slice(:, 1),
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()