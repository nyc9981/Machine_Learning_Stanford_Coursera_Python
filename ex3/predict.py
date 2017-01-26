import numpy as np

from ex2.sigmoid import sigmoid
 

def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

# Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape
    
    # You need to return the following variables correctly
    p = np.zeros((m, 1))

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
#
    
    
    # Add ones to the X data matrix
    a1 = np.column_stack((np.ones((m, 1)), X))
    
    # second layer (note: add ones the matrix)
    a2 = np.column_stack((np.ones((m, 1)), sigmoid(np.dot(a1, Theta1.T))))
    
    a3 = sigmoid(np.dot(a2, Theta2.T))
    
    # take the indexes of max prob per row
    p = np.argmax(a3, axis=1)
        
# =========================================================================

    return p + 1        # add 1 to offset index of maximum in A row

