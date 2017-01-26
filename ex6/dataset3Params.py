import numpy as np
from sklearn import svm
from itertools import product


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

# You need to return the following variables correctly.
    C = 1 # place holder
    sigma = 0.3 # place holder

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example, 
#                   predictions = svmPredict(model, Xval)
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using 
#        mean(double(predictions ~= yval))
#
    best_accuracy = -1  # 
    para_list = [.01, .03, .1, .3, 1., 3., 10., 30.]
    C_sigma_comb = product(para_list, para_list)
    
    for C_i, sigma_i in C_sigma_comb:
        # Train the SVM
        gamma = 1.0 / (2.0 * sigma_i ** 2)
        clf = svm.SVC(C=C_i, kernel='rbf', tol=1e-3, max_iter=500, gamma=gamma)
        model = clf.fit(X, y)
        # Predict
        pred = model.predict(Xval)
        # prediction accuracy
        pred_accuracy = np.mean(pred == yval)
        # if prediction accuracy is larger, then record C and sigma
        #print C_i, sigma_i, pred_accuracy
        if pred_accuracy > best_accuracy:
            best_accuracy = pred_accuracy
            C, sigma = C_i, sigma_i
            
# =========================================================================
    return C, sigma
