from numpy import asfortranarray, squeeze, asarray

from gradientFunction import gradientFunction
from sigmoid import sigmoid
import numpy as np


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)   # number of training examples
    #n = X.shape[1]
    #grad = np.zeros(n)

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
    
    h = sigmoid(X.dot(theta)) # m by 1
    
    # unregularized gradient
    grad = X.T.dot(h-y) / (1.0 * m) # n by 1
    
    theta1 = theta.copy()
    theta1[0] = 0 # theta0 is not regularized
    
    grad_reg_term = 1.0 * Lambda * theta1 / m
    
    grad = grad + grad_reg_term

# =============================================================

    return grad