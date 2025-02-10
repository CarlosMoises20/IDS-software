
from pyspark.sql.functions import col, when, count, explode, expr
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors        # for kNN
from sklearn import model_selection
import os
from intrusion_detection import *
from crate.client import connect
import pandas as pd
from data_preprocessing import *


### On this module, add functions, where each function process a different type of messages


# Auxiliary function to bind all log files inside a indicated directory
# into a single log file of each type of LoRaWAN message
def bind_dir_files(dataset_path, output_filename):

    # Skip file generation if it already exists
    if os.path.exists(output_filename):
        print(f"File '{output_filename}' already exists. Skipping generation.")
        return

    all_logs = []                         # Create a list to store the different files

    for filename in dataset_path:
        with open(filename, 'r') as f:
            all_logs.append(f.read())     # Append the contents of the file to the list

    # Join all logs into a single string
    combined_logs = '\n'.join(all_logs)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write the combined logs to a new file
    with open(output_filename, 'w') as f:
        f.write(combined_logs)




def process_rxpk_dataset(spark_session, dataset):

    ### Bind all log files into a single log file to simplify data processing
    
    combined_logs_filename = './combined_datasets/combined_rxpk_logs.log'
    bind_dir_files(dataset, combined_logs_filename)


    # Load the dataset into a Spark Dataframe
    df = spark_session.read.json(combined_logs_filename)


    ### Pre-Processing
    df = pre_process_rxpk_dataset(df)


    #knn = KNeighborsClassifier(n_neighbors=10, algorithm='auto')
    #knn.fit(df.rxpk.rssi, df.rxpk.snr)

    # Add anomaly detection columns
    df = df.withColumn("Jamming", when(jamming_detection(df.rxpk.rssi), 1).otherwise(0))

    # Combine anomaly indicators (TODO: juntar com todas as outras anomalias)
    #df = df.withColumn("Anomaly", (col("Jamming")) > 0)

    # Group data for analysis
    """
    summary = df.groupBy("Anomaly").agg(
        count("*").alias("MessageCount"),
        count(when(col("Anomaly_SNR") == 1, 1)).alias("SNR_Anomalies"),
        count(when(col("Anomaly_RSSI") == 1, 1)).alias("RSSI_Anomalies"),
        count(when(col("Anomaly_MIC") == 1, 1)).alias("MIC_Anomalies"),
        count(when(col("Anomaly_Size") == 1, 1)).alias("Size_Anomalies")
    )
    """


    pass




def process_txpk_dataset(spark_session, dataset):

    ### Bind all log files into a single log file

    combined_logs_filename = './combined_datasets/combined_txpk_logs.log'
    bind_dir_files(dataset, combined_logs_filename)

    # Load the dataset into a Spark Dataframe
    df = spark_session.read.json(combined_logs_filename)

    ### Pre-Processing
    df = pre_process_txpk_dataset(df)

    pass


