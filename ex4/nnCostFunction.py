import numpy as np
from numpy import log
from ex2.sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):

    """computes the cost and gradient of the neural network. The
  parameters for the neural network are "unrolled" into the vector
  nn_params and need to be converted back into the weight matrices.

  The returned parameter grad should be a "unrolled" vector of the
  partial derivatives of the neural network.
    """

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
# Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                       (hidden_layer_size, input_layer_size + 1), order='F').copy()

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                       (num_labels, (hidden_layer_size + 1)), order='F').copy()



# Setup some useful variables
    m, _ = X.shape


# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the code by working through the
#               following parts.
#
# Part 1: Feedforward the neural network and return the cost in the
#         variable J. After implementing Part 1, you can verify that your
#         cost function computation is correct by verifying the cost
#         computed in ex4.m
#
# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#
#         Note: The vector y passed into the function is a vector of labels
#               containing values from 1..K. You need to map this vector into a 
#               binary vector of 1's and 0's to be used with the neural network
#               cost function.
#
#         Hint: We recommend implementing backpropagation using a for-loop
#               over the training examples if you are implementing it for the 
#               first time.
#
# Part 3: Implement regularization with the cost function and gradients.
#
#         Hint: You can implement this around the code for
#               backpropagation. That is, you can compute the gradients for
#               the regularization separately and then add them to Theta1_grad
#               and Theta2_grad from Part 2.
#  
    # Expand y to matrix Y
    Y = np.zeros((m, num_labels)) # m X num_labels        
    for i in range(m):
        j = y[i]-1 # index = label - 1
        Y[i, j] = 1
    
    # Feed forward
    a1 = np.column_stack((np.ones((m, 1)), X)) # m X (input_layer_size + 1)
    z2 = a1.dot(Theta1.T) # m X hidden_layer_size
    a2 = np.column_stack((np.ones((m, 1)), sigmoid(z2))) # m X hidden_layer_size + 1
    z3 = a2.dot(Theta2.T)   
    a3 = sigmoid(z3) # m X num_labels
    h = a3 # m X num_labels
    
    # Set the first column of regularized version of both Theta1 and Theta2 to 0,
    # because the first term of Theta does not get regularized.
    # Used later to calculate regularized terms of cost function and gradient function   
    Theta1_reg = Theta1.copy()
    Theta1_reg[:,0] = 0
    Theta2_reg = Theta2.copy()
    Theta2_reg[:,0] = 0 
    
    # Calculte regularization term
    # Note: Theta1_reg and Theta1_reg are used to calculate reg terms
    r = np.sum(Theta1_reg**2) * Lambda * 0.5 / m + np.sum(Theta2_reg**2) * Lambda * 0.5 / m
    
    # Regularized cost
    J = np.sum(Y * log(h) + (1-Y) * log(1-h)) / (-m) + r

    # =========================================================================
    Theta1_grad = np.zeros(Theta1.shape) # hidden_layer_size X input_layer_size + 1
    Theta2_grad = np.zeros(Theta2.shape) # num_labels X hidden_layer_size + 1
    
    ## Backward propagation
    sigma3 = a3 - Y # m X num_labels
    sigma2 = sigma3.dot(Theta2)[:, 1:] * sigmoidGradient(z2) # m X hidden_layer_size
    # Accumulate the errors, then divide by m,  then add the reg terms
    # Note: Theta1_reg and Theta1_reg are used to calculate reg terms
    Theta1_grad = sigma2.T.dot(a1) / m + Lambda * Theta1_reg / m
    Theta2_grad = sigma3.T.dot(a2) / m + Lambda * Theta2_reg / m
      
    # Unroll gradient
    grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))

    return J, grad