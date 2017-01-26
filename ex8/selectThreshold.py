import numpy as np
import math

def selectThreshold(yval, pval):
    """
    finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    bestEpsilon = 0
    bestF1 = 0

    stepsize = (np.max(pval) - np.min(pval)) / 1000.0
    for epsilon in np.arange(np.min(pval),np.max(pval), stepsize):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #
        # Note: You can use predictions = (pval < epsilon) to get a binary vector
        #       of 0's and 1's of the outlier predictions
        
        pred = pval < epsilon
        true_positive = np.sum( (yval==1) & (pred==1) )
        true_negative = np.sum( (yval==0) & (pred==0) )
        false_positive = np.sum( (yval==0) & (pred==1) )
        false_negative = np.sum( (yval==1) &  (pred==0) )
        #print true_positive, true_negative, false_positive, false_negative
        #if (true_positive + false_positive) != 0 and (true_positive + false_negative) != 0:
        try:
            precision = 1.0 * true_positive / (true_positive + false_positive)
            recall = 1.0 * true_positive / (true_positive + false_negative)
            F1 = 2.0 * precision * recall / (precision + recall)

        # =============================================================

            if F1 > bestF1:
               bestF1 = F1
               bestEpsilon = epsilon
        except Exception:
            pass

    return bestEpsilon, bestF1






