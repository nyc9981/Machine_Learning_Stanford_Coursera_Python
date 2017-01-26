import numpy as np
def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m = y.size # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear 
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
#


# =========================================================================
    
    h = X.dot(theta)
    
    # Note: do not regularize the first term of theta
    theta_reg = theta.copy()
    theta_reg[0] = 0
    
    J = (h-y).dot(h-y) / (2.0*m)+ Lambda * np.sum(theta_reg ** 2) / (2.0*m)
    
    # Add 1.0 to avoid integer division
    grad = X.T.dot(h-y) / (1.0*m) + Lambda * theta_reg / (1.0*m)
    
    return J, grad