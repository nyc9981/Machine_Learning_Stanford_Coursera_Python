import numpy as np

def randInitializeWeights(L_in, L_out):
    """randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing
      connections.

      Note that W should be set to a matrix of size(L_out, 1 + L_in) as the column row of W handles the "bias" terms
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Initialize W randomly so that we break the symmetry while
    #               training the neural network.
    #
    # Note: The first row of W corresponds to the parameters for the bias units
    #
    
    epi = .12
    W = np.random.uniform(size=(L_out, 1 + L_in)) * epi * 2 - epi


# =========================================================================

    return W
