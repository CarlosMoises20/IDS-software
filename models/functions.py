
import numpy as np
from scipy.spatial.distance import hamming
from scipy.stats import entropy


# These functions might be used in the future by another algorithms, but they 
# won't be used directly


"""
Function For Finding Euclidean Distance Between Two Rows

The function returns None if at least one of the rows is None

"""
def euclidean_dist(row1, row2):
    dist = 0
    for entry in range(len(row1)-1):
        if row1[entry] == None or row2[entry] == None:
            return 1e10
        dist += (float(row1[entry]) - float(row2[entry])) ** 2 
    return dist


"""
This function computes the hamming distance between two given strings
that must have the same length

The function returns None if at least one of the strings is None or 
if the strings have different lengths

"""
def hamming_distance(str1, str2):
    
    if (str1 is None) or (str2 is None):
        return None
    
    if len(str1) != len(str2):
        return None
    
    return hamming(list(str1), list(str2))


"""
This function computes the Kullback-Leibler Divergence between two given probability distributions

P is the baseline probability distribution, the learnt distribution
Q is the new probability distribution, the real-time distribution of the random number generated in the 
    Join-Request

"""
def kl_divergence(P, Q):

    P = np.array(P, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)

    # Avoid division by zero, replacing zero elements with a very small value (??)
    P[P == 0] = 1e-10
    Q[Q == 0] = 1e-10
    
    return entropy(P, Q, base=2)  # Using log base 2 for better interpretability


# Sigmoid Activation function
def sigmoid(X):
    return 1 / (1 + np.exp(-X))