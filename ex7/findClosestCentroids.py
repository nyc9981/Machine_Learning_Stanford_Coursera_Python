import numpy as np


def findClosestCentroids(X, centroids):
    """returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    """

# Set K, ie number of rows
    K = len(centroids)

# You need to return the following variables correctly.
    m = X.shape[0]
    idx = np.zeros(m, dtype=np.int)
    val = np.zeros(m, dtype=np.int)

# ====================== YOUR CODE HERE ======================
# Instructions: Go over every example, find its closest centroid, and store
#               the index inside idx at the appropriate location.
#               Concretely, idx(i) should contain the index of the centroid
#               closest to example i. Hence, it should be a value in the 
#               range 1..K
#
# Note: You can use a for-loop over the examples to compute this.
    
    for i in range(m):
        d = np.array([np.sum((X[i, :] - c) ** 2) for  c in centroids])
        i_c = np.argmin(d)
        idx[i] = i_c    # index
        val[i]= i_c + 1 # label
    
# =============================================================

    return val, idx

