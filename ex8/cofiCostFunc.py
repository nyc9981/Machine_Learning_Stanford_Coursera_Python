import numpy as np


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda):
    """returns the cost and gradient for the
    """

    # Unfold the U and W matrices from params
    X = np.array(params[:num_movies*num_features]).reshape(num_features, num_movies).T.copy()
    Theta = np.array(params[num_movies*num_features:]).reshape(num_features, num_users).T.copy()


    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        X_grad - num_movies x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of X
    #        Theta_grad - num_users x num_features matrix, containing the
    #                     partial derivatives w.r.t. to each element of Theta
    
    d = np.zeros((Y.shape))
    d = X.dot(Theta.T) - Y
    d[R==0] = 0 # set to 0 where there is not rating

    J = np.sum(d**2) / 2.0 # cost w/o regularization
    J_reg_term = Lambda / 2.0 * (np.sum(Theta**2) + np.sum(X**2))
    J = J + J_reg_term
    
    X_grad = d.dot(Theta) # X gradient w/o regularization
    X_grad_reg_term = Lambda * X
    X_grad = X_grad + X_grad_reg_term
    
    Theta_grad = d.T.dot(X) # Theta gradient w/o regularization
    Theta_grad_reg_term = Lambda * Theta
    Theta_grad = Theta_grad + Theta_grad_reg_term

    # =============================================================

    grad = np.hstack((X_grad.T.flatten(),Theta_grad.T.flatten()))

    return J, grad