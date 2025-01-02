
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors        # for kNN
from constants import *



### On this module, add functions, where each function represents one or more types of intrusions



## Gateway and device changes (kNN usage)
def device_gateway_analysis(df):


    # TODO: implement this  

    vectorized_data = df.select(df["rssi"], df["lsnr"], df["tmst"], df["len"]).collect()

    # Create a kNN model
    model = KNeighborsClassifier(n_neighbors=10, algorithm='auto')

    model.fit(vectorized_data)

    model.predict(vectorized_data)

    
    pass