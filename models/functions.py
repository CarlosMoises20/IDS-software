
import numpy as np

# Sigmoid Activation function
def sigmoid(X):
    return 1 / (1 + np.exp(-X))