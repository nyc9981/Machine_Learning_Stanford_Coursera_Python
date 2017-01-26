import numpy as np


def computeCentroids(X, idx, K):
    """returns the new centroids by
    computing the means of the data points assigned to each centroid. It is
    given a dataset X where each row is a single data point, a vector
    idx of centroid assignments (i.e. each entry in range [1..K]) for each
    example, and K, the number of centroids. You should return a matrix
    centroids, where each row of centroids is the mean of the data points
    assigned to it.
    """

# Useful variables
    m, n = X.shape

# You need to return the following variables correctly.
    centroids = []


# ====================== YOUR CODE HERE ======================
# Instructions: Go over every centroid and compute mean of all points that
#               belong to it. Concretely, the row vector centroids(i, :)
#               should contain the mean of the data points assigned to
#               centroid i.
#
# Note: You can use a for-loop over the centroids to compute this.
# 
    
    for k in range(K):
        #X_k = X[np.where(idx==k)]
        X_k = X[idx==k]
        cen_k = np.mean(X_k, axis=0)
        centroids.append(cen_k)


# =============================================================

    return centroids
