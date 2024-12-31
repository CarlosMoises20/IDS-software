
from sklearn.neighbors import KNeighborsClassifier        # for kNN
from constants import *



### On this module, add functions, where each function represents one or more types of intrusions



## Gateway and device changes (kNN usage)
def device_gateway_analysis(df):

    # Create a kNN model
    model = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    pass