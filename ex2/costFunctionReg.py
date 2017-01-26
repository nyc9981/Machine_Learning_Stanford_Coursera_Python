from costFunction import costFunction
# import numpy as np
# from sigmoid import sigmoid

def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

    theta1 = theta.copy()
    theta1[0] = 0 # theta0 is not regularized
    J = costFunction(theta, X, y) + Lambda * theta1.dot(theta1) / (2.0 * m)
# =============================================================

    return J
