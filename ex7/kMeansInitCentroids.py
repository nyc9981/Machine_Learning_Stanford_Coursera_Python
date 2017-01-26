import numpy as np


def kMeansInitCentroids(X, K):
    """returns K initial centroids to be
    used with the K-Means on the dataset X
    """

# You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))

# ====================== YOUR CODE HERE ======================
# Instructions: You should set centroids to randomly chosen examples from
#               the dataset X
#
    rand_row_idx = np.random.choice(X.shape[0], K, replace=False)
    centroids = X[rand_row_idx,:]

# =============================================================
    return centroids
