
from pyspark.sql.functions import col, when, count, explode
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors        # for kNN
from sklearn import model_selection
import os

### On this module, add functions, where each function process a different type of messages


# Auxiliary function to bind all log files into a single log file of each type of LoRaWAN message
def bind_dir_files(dataset_path, output_filename):

    ### Bind all log files into a single log file

    all_logs = []

    for filename in dataset_path:
        with open(filename, 'r') as f:
            all_logs.append(f.read())     # Append the contents of the file to the list


    # Join all logs into a single string
    combined_logs = '\n'.join(all_logs)

    # Write the combined logs to a new file
    with open(output_filename, 'w') as f:
        f.write(combined_logs)



# 1 - Converts a dataset of type 'rxpk', given the filename of the dataset, into a 'df' Spark dataframe
# 2 - Applies feature selection techniques to remove the most irrelevant attributes (dimensionality reduction),
#        selecting only the attributes that are relevant to build the intended model for IDS 
def pre_process_rxpk_dataset(spark_session, filename):

    # Load the data from the dataset file
    df = spark_session.read.json(filename)

    # Explode the 'rxpk' array
    df = df.withColumn("rxpk", explode(col("rxpk")))

    # Extract individual fields from 'rxpk'
    for field in df.schema["rxpk"].dataType.fields:
        df = df.withColumn(field.name, col(f"rxpk.{field.name}"))

    # Drop the 'rxpk' column as it's now flattened
    df = df.drop("rxpk")

    # Explode the 'rsig' array
    df = df.withColumn("rsig", explode(col("rsig")))

    # Extract individual fields from 'rsig'
    for field in df.schema["rsig"].dataType.fields:
        df = df.withColumn(field.name, col(f"rsig.{field.name}"))

    # Drop the 'rsig' column as it's now flattened
    df = df.drop("rsig")


    ## TODO: filter attributes (2)
    df = df.drop("type", "totalrxpk")

    return df




# 1 - Converts a dataset of type 'txpk', given the filename of the dataset, into a 'df' Spark dataframe
# 2 - Applies feature selection techniques to remove the most irrelevant attributes (dimensionality reduction),
#        selecting only the attributes that are relevant to build the intended model for IDS 
def pre_process_txpk_dataset(spark_session, filename):

    # Load the data from the dataset file
    df = spark_session.read.json(filename)

    # Extract individual fields from 'txpk'
    for field in df.schema["txpk"].dataType.fields:
        df = df.withColumn(field.name, col(f"txpk.{field.name}"))

    # Drop the 'txpk' column as it's now flattened
    df = df.drop("txpk")

    ## TODO: filter attributes (2)
    df = df.drop("type")

    return df




def process_rxpk_dataset(spark_session, dataset):

    ### Bind all log files into a single log file
    
    combined_logs_filename = './combined_datasets/combined_rxpk_logs.log'
    bind_dir_files(dataset, combined_logs_filename)


    ### Pre-Processing
    
    df = pre_process_rxpk_dataset(spark_session, combined_logs_filename)


    pass




def process_txpk_dataset(spark_session, dataset):

    ### Bind all log files into a single log file

    combined_logs_filename = './combined_datasets/combined_txpk_logs.log'
    bind_dir_files(dataset, combined_logs_filename)


    ### Pre-Processing
    
    df = pre_process_txpk_dataset(spark_session, combined_logs_filename)

    pass





"""
vectorized_data = [.....]

# Create a kNN model
model = KNeighborsClassifier(n_neighbors=10, algorithm='auto')

model.fit(vectorized_data)

model.predict(vectorized_data)

"""